//! jevhome: runtime for the typed-decision models (a port of the Python inference path, same ONNX graphs).
//!
//! A model folder (checkpoints/onnx-slim/<model>/) holds tokenizer.json, tokenizer_config.json,
//! decision_config.json, calibration_u.json and either model.onnx (cross-encoder, "h2") or
//! encoder.onnx + scorer.onnx (bi-encoder, "emb"). Everything model-specific is read from those files.
//!
//! Ported from Python: SemIfDirectAdapter.build_request (options), decisions.render._h2_build (h2 layout),
//! decisions.embed_model.texts_for/build_batch (emb inputs), bench_baseline.to_probs (temperature softmax).
pub mod serve;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

const MASK_ID: i64 = 50284;
const CLS_ID: i64 = 50281;
const SEP_ID: i64 = 50282;
const H2_MAX_LEN: usize = 512;
const H2_HEAD_MAX_LEN: usize = 320;
const H2_OPT_MAX: usize = 48;
const EMB_MAX_STATE: usize = 512;
const EMB_MAX_QUERY: usize = 160;

/// One option of a canonical request: id and "id: description".
pub struct Opt {
    pub id: String,
    pub description: String,
}

/// A task as read from a JevBench-style JSON row (state, question {type, instructions, criteria}).
pub struct Request {
    pub qtype: String,
    pub question: String,
    pub state: String,
    pub options: Vec<Opt>,
}

impl Request {
    /// SemIfDirectAdapter.build_request: noul = [true, false], choice = criteria order, score = "i: level".
    pub fn from_row(row: &Value) -> Result<Self> {
        Self::new(&row["state"], &row["question"])
    }

    /// One question ({type, instructions, criteria}) about a state (string or any JSON value).
    pub fn new(state: &Value, q: &Value) -> Result<Self> {
        let qtype = q["type"].as_str().ok_or_else(|| anyhow!("question.type missing"))?.to_string();
        let crit = &q["criteria"];
        let mut options: Vec<Opt> = match qtype.as_str() {
            "noul" => ["true", "false"]
                .iter()
                .map(|k| Opt {
                    id: k.to_string(),
                    description: crit.get(*k).and_then(Value::as_str).map(str::to_string).unwrap_or(format!("The proposition is {k}.")),
                })
                .collect(),
            "choice" => crit
                .as_object()
                .ok_or_else(|| anyhow!("choice criteria must be an object"))?
                .iter()
                .map(|(k, v)| {
                    let d = v.as_str().unwrap_or("");
                    Opt { id: k.clone(), description: if d.is_empty() { k.clone() } else { d.to_string() } }
                })
                .collect(),
            "score" => crit
                .as_array()
                .ok_or_else(|| anyhow!("score criteria must be a list"))?
                .iter()
                .enumerate()
                .map(|(i, v)| Opt { id: i.to_string(), description: py_str(v) })
                .collect(),
            other => bail!("question type must be noul, choice or score, not {other:?}"),
        };
        for o in options.iter_mut() {
            o.description = format!("{}: {}", o.id, o.description);
        }
        Ok(Request {
            qtype,
            question: py_str(&q["instructions"]),
            state: state_text(state),
            options,
        })
    }
}

/// Python str() of a JSON value that is normally a string.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_json(other),
    }
}

/// decisions.render.state_text: strings as-is, anything else as json.dumps(ensure_ascii=False).
pub fn state_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_json(other),
    }
}

/// json.dumps(v, ensure_ascii=False) with Python's default separators (", ", ": ") and float repr.
pub fn py_json(v: &Value) -> String {
    let mut out = String::new();
    write_py_json(v, &mut out);
    out
}

fn write_py_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string())
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string())
            } else {
                out.push_str(&py_float(n.as_f64().unwrap()))
            }
        }
        Value::String(s) => write_py_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_str(k, out);
                out.push_str(": ");
                write_py_json(x, out);
            }
            out.push('}');
        }
    }
}

fn write_py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python float repr: shortest round-trip digits, scientific notation when exponent < -4 or >= 16.
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let sci = format!("{:e}", f); // e.g. "-1.25e-5", shortest round-trip digits
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let s = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{sign}{m}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs())
    }
}

fn qtype_id(q: &str) -> i64 {
    match q {
        "choice" => 0,
        "score" => 1,
        _ => 2,
    }
}

/// decisions.adapters.bucket
fn bucket(qtype: &str, k: usize) -> String {
    let size = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
    format!("{qtype}:{size}")
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("{e}"))?
        .with_inter_threads(1)
        .map_err(|e| anyhow!("{e}"))?
        .with_parallel_execution(false)
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))?)
}

enum Graphs {
    H2 { model: Session },
    Emb { encoder: Session, scorer: Session },
}

/// Model inputs of one decision, kept for comparison with the Python token ids.
pub struct Encoded {
    pub ids: Vec<i64>,          // h2: the full sequence; emb: the state
    pub query_ids: Vec<Vec<i64>>, // emb: one query per option (unpadded)
}

pub struct Model {
    graphs: Graphs,
    tok: Tokenizer,
    tok_query: Tokenizer,
    tok_plain: Tokenizer,
    mask_token: String,
    temps: HashMap<String, f32>,
}

impl Model {
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        let read = |f: &str| -> Result<Value> {
            let p: PathBuf = dir.join(f);
            Ok(serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?)?)
        };
        let cfg = read("decision_config.json")?;
        let head = cfg["head"].as_str().unwrap_or("");
        let graphs = match head {
            "h2" => {
                if cfg["head_layers"].as_u64().unwrap_or(0) != 0 || cfg["typed"].as_bool().unwrap_or(false) || cfg["ordinal"].as_bool().unwrap_or(false) {
                    bail!("h2 variant not supported: {cfg}");
                }
                Graphs::H2 { model: session(&dir.join("model.onnx"), threads)? }
            }
            "emb" => {
                if cfg["variant"] != "bi" || cfg["pooling"] != "cls" {
                    bail!("emb variant not supported: {cfg}");
                }
                // the scorer is a tiny MLP: 1 thread, so no second thread pool competes with the encoder's for the cores
                Graphs::Emb { encoder: session(&dir.join("encoder.onnx"), threads)?, scorer: session(&dir.join("scorer.onnx"), 1)? }
            }
            _ => bail!("unknown head {head}"),
        };
        let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        tok.with_padding(None);
        let mut tok_query = tok.clone();
        let mut tok_plain = tok.clone(); // no truncation: the h2 pieces are truncated by hand
        tok_plain.with_truncation(None).map_err(|e| anyhow!("{e}"))?;
        // HF: tok(state, truncation=True, max_length=512) / tok(instr, crit, truncation="longest_first", max_length=160)
        tok.with_truncation(Some(TruncationParams {
            max_length: EMB_MAX_STATE,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }))
        .map_err(|e| anyhow!("{e}"))?;
        tok_query
            .with_truncation(Some(TruncationParams {
                max_length: EMB_MAX_QUERY,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| anyhow!("{e}"))?;
        tok_query.with_padding(Some(PaddingParams { strategy: PaddingStrategy::BatchLongest, ..Default::default() }));
        let tcfg = read("tokenizer_config.json")?;
        let mask_token = tcfg["mask_token"].as_str().unwrap_or("[MASK]").to_string();
        let cal = read("calibration_u.json")?;
        let temps = cal
            .as_object()
            .map(|o| {
                o.iter()
                    .filter(|(k, _)| !k.starts_with('_'))
                    .filter_map(|(k, v)| v.get("T").unwrap_or(v).as_f64().map(|t| (k.clone(), t as f32)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Model { graphs, tok, tok_query, tok_plain, mask_token, temps })
    }

    fn ids_plain(&self, text: &str) -> Result<Vec<i64>> {
        let e = self.tok_plain.encode(text, false).map_err(|e| anyhow!("{e}"))?;
        Ok(e.get_ids().iter().map(|&x| x as i64).collect())
    }

    /// decisions.render._h2_build: [CLS] head [SEP] ([MASK] opt)* [SEP] state [SEP] -> (ids, marker positions).
    fn h2_ids(&self, r: &Request) -> Result<(Vec<i64>, Vec<i64>)> {
        let m = &self.mask_token;
        let ins = r.question.replace(m.as_str(), " ");
        let mut head = self.ids_plain(&format!("{} question: {}", r.qtype, ins))?;
        let mut opts: Vec<Vec<i64>> = Vec::new();
        for o in &r.options {
            let mut t = self.ids_plain(&format!(" {}", o.description.replace(m.as_str(), " ")))?;
            t.truncate(H2_OPT_MAX);
            let mut v = vec![MASK_ID];
            v.extend(t);
            opts.push(v);
        }
        let total = |opts: &Vec<Vec<i64>>| opts.iter().map(Vec::len).sum::<usize>() as i64;
        let mut budget = H2_HEAD_MAX_LEN as i64 - total(&opts);
        if budget < 16 {
            let per = std::cmp::max(4, (H2_HEAD_MAX_LEN as i64 - 16) / std::cmp::max(1, opts.len() as i64)) as usize;
            for o in opts.iter_mut() {
                o.truncate(per);
            }
            budget = H2_HEAD_MAX_LEN as i64 - total(&opts);
        }
        head.truncate(std::cmp::max(8, budget).max(0) as usize);
        let mut ids = vec![CLS_ID];
        ids.extend(head);
        ids.push(SEP_ID);
        let mut markers = Vec::new();
        for o in opts {
            markers.push(ids.len() as i64);
            ids.extend(o);
        }
        ids.push(SEP_ID);
        let st = self.ids_plain(&r.state.replace(m.as_str(), " "))?;
        let room = (H2_MAX_LEN as i64 - ids.len() as i64 - 1).max(0) as usize;
        ids.extend(st.into_iter().take(room));
        ids.push(SEP_ID);
        ids.truncate(H2_MAX_LEN);
        Ok((ids, markers))
    }

    /// Emb only: the state vector (tok(state, truncation=True, max_length=512) -> encoder), and its token ids.
    fn encode_state(&mut self, state: &str) -> Result<StateVec> {
        let se = self.tok.encode(state, true).map_err(|e| anyhow!("{e}"))?;
        let ids: Vec<i64> = se.get_ids().iter().map(|&x| x as i64).collect();
        let Graphs::Emb { encoder, .. } = &mut self.graphs else { bail!("not a bi-encoder") };
        let n = ids.len();
        let o = encoder.run(ort::inputs![
            "ids" => Tensor::from_array(([1usize, n], ids.clone()))?,
            "att" => Tensor::from_array(([1usize, n], vec![1i64; n]))?,
        ])?;
        let (shape, v) = o["vec"].try_extract_tensor::<f32>()?;
        Ok(StateVec { ids, dim: shape[1] as usize, vec: v.to_vec() })
    }

    /// Raw logits (canonical option order) of one request; emb models take the already encoded state.
    fn logits(&mut self, r: &Request, state: Option<&StateVec>) -> Result<(Vec<f32>, Encoded)> {
        let k = r.options.len();
        if matches!(self.graphs, Graphs::H2 { .. }) {
            let (ids, markers) = self.h2_ids(r)?;
            let n = ids.len();
            let Graphs::H2 { model } = &mut self.graphs else { unreachable!() };
            let out = model.run(ort::inputs![
                "input_ids" => Tensor::from_array(([1usize, n], ids.clone()))?,
                "attention_mask" => Tensor::from_array(([1usize, n], vec![1i64; n]))?,
                "marker_pos" => Tensor::from_array(([1usize, markers.len()], markers))?,
                "qtype" => Tensor::from_array(([1usize], vec![qtype_id(&r.qtype)]))?,
            ])?;
            let (_, l) = out["logits"].try_extract_tensor::<f32>()?;
            return Ok((l[..k].to_vec(), Encoded { ids, query_ids: vec![] }));
        }
        // queries: tok(instr, crit) pairs, longest_first 160, padded to the longest
        let s = state.ok_or_else(|| anyhow!("bi-encoder needs the encoded state"))?;
        let instr = format!("{} question: {}", r.qtype, r.question);
        let pairs: Vec<(String, String)> = r.options.iter().map(|o| (instr.clone(), o.description.clone())).collect();
        let qe = self.tok_query.encode_batch(pairs, true).map_err(|e| anyhow!("{e}"))?;
        let ql = qe[0].get_ids().len();
        let qids: Vec<i64> = qe.iter().flat_map(|e| e.get_ids().iter().map(|&x| x as i64)).collect();
        let qatt: Vec<i64> = qe.iter().flat_map(|e| e.get_attention_mask().iter().map(|&x| x as i64)).collect();
        let unpadded: Vec<Vec<i64>> = qe
            .iter()
            .map(|e| e.get_ids().iter().zip(e.get_attention_mask()).filter(|(_, &m)| m == 1).map(|(&x, _)| x as i64).collect())
            .collect();
        let Graphs::Emb { encoder, scorer } = &mut self.graphs else { unreachable!() };
        let q = {
            let o = encoder.run(ort::inputs![
                "ids" => Tensor::from_array(([k, ql], qids))?,
                "att" => Tensor::from_array(([k, ql], qatt))?,
            ])?;
            let (_, v) = o["vec"].try_extract_tensor::<f32>()?;
            v.to_vec()
        };
        let srep: Vec<f32> = (0..k).flat_map(|_| s.vec.iter().copied()).collect();
        let o = scorer.run(ort::inputs![
            "s" => Tensor::from_array(([k, s.dim], srep))?,
            "q" => Tensor::from_array(([k, s.dim], q))?,
        ])?;
        let (_, l) = o["logits"].try_extract_tensor::<f32>()?;
        Ok((l.to_vec(), Encoded { ids: s.ids.clone(), query_ids: unpadded }))
    }

    /// bench_baseline.to_probs: softmax of logits / T(bucket).
    fn calibrate(&self, r: &Request, logits: &[f32]) -> Vec<f32> {
        let t = *self.temps.get(&bucket(&r.qtype, r.options.len())).unwrap_or(&1.0);
        let z: Vec<f32> = logits.iter().map(|x| x / t).collect();
        let mx = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = z.iter().map(|x| (x - mx).exp()).collect();
        let sum: f32 = e.iter().sum();
        e.iter().map(|x| x / sum).collect()
    }

    pub fn is_emb(&self) -> bool {
        matches!(self.graphs, Graphs::Emb { .. })
    }

    /// Calibrated probabilities (canonical option order) for one task row, plus the model inputs.
    pub fn decide(&mut self, row: &Value) -> Result<(Request, Vec<f32>, Encoded)> {
        let r = Request::from_row(row)?;
        let s = if self.is_emb() { Some(self.encode_state(&r.state)?) } else { None };
        let (l, enc) = self.logits(&r, s.as_ref())?;
        let p = self.calibrate(&r, &l);
        Ok((r, p, enc))
    }

    /// Several questions about one state: each question is one decision; a bi-encoder encodes the state once.
    /// Returns per question the request, the calibrated probabilities and the tokens it fed to the model
    /// (the state's tokens are counted once).
    pub fn decide_many(&mut self, state: &Value, questions: &[&Value]) -> Result<Vec<(Request, Vec<f32>, usize)>> {
        let reqs: Vec<Request> = questions.iter().map(|q| Request::new(state, q)).collect::<Result<_>>()?;
        let s = match (self.is_emb(), reqs.first()) {
            (true, Some(r)) => Some(self.encode_state(&r.state)?),
            _ => None,
        };
        let mut out = Vec::with_capacity(reqs.len());
        for (i, r) in reqs.into_iter().enumerate() {
            let (l, enc) = self.logits(&r, s.as_ref())?;
            let p = self.calibrate(&r, &l);
            let q: usize = enc.query_ids.iter().map(Vec::len).sum();
            let n = if s.is_none() { enc.ids.len() } else if i == 0 { enc.ids.len() + q } else { q };
            out.push((r, p, n));
        }
        Ok(out)
    }
}

/// An encoded state (bi-encoder): token ids and the pooled vector.
struct StateVec {
    ids: Vec<i64>,
    dim: usize,
    vec: Vec<f32>,
}

impl Model {
    /// Number of tokens of a text (no special tokens), e.g. for the usage block of the HTTP API.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.tok_plain.encode(text, false).map(|e| e.get_ids().len()).unwrap_or(0)
    }
}
