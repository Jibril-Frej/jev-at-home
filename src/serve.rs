//! HTTP API compatible with Jev:
//! POST /v1/systemone {state, questions: {name: {type, instructions, criteria}}, model?, options?}
//!   -> {model, answers: {name: answer}, usage, latency_ms}
//! GET /v1/models -> {models: [...]}
//! Same validation, answer fields, rounding and error codes (422 {"detail"}; 404 unknown path). The reasoning
//! options are accepted and ignored: these models answer in one forward pass per question, without reasoning.
//! One request at a time (the model is not shared between threads).
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};
use tiny_http::{Header, Method, Response, Server};

use crate::{py_json, Model, Request};

const MAX_OPTIONS: usize = 255;

struct Args {
    dir: String,
    host: String,
    port: u16,
    threads: usize,
}

fn parse_args(a: &[String]) -> Result<Args> {
    let usage = "usage: jevhome serve <model_dir> [--port 8009] [--host 127.0.0.1] [--threads 4]";
    let mut args = Args { dir: String::new(), host: "127.0.0.1".into(), port: 8009, threads: 4 };
    let mut it = a.iter();
    while let Some(x) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| anyhow!("{x} needs a value\n{usage}"));
        match x.as_str() {
            "--port" => args.port = val()?.parse()?,
            "--host" => args.host = val()?,
            "--threads" => args.threads = val()?.parse()?,
            s if s.starts_with("--") => return Err(anyhow!("unknown option {s}\n{usage}")),
            s => args.dir = s.to_string(),
        }
    }
    if args.dir.is_empty() {
        return Err(anyhow!("{usage}"));
    }
    if args.threads == 0 {
        return Err(anyhow!("--threads must be at least 1"));
    }
    Ok(args)
}

fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// Round to 2 decimals, from the exact binary value.
fn r2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap()
}

/// Quoted string / list of strings for error messages: 'a', ['a', 'b'].
fn pyr(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn pyr_list(v: &[&str]) -> String {
    format!("[{}]", v.iter().map(|s| pyr(s)).collect::<Vec<_>>().join(", "))
}

/// Index of the first maximum (ties go to the first option).
fn argmax(p: &[f64]) -> usize {
    (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b })
}

/// Request options: only these keys, with these types. The values change nothing here.
fn check_options(raw: &Value) -> Result<(), String> {
    if raw.is_null() {
        return Ok(());
    }
    let o = raw.as_object().ok_or("options must be an object")?;
    let mut unknown: Vec<&str> = o.keys().map(String::as_str).filter(|k| !["think", "max_think", "nothink_threshold", "return_reasoning"].contains(k)).collect();
    if !unknown.is_empty() {
        unknown.sort();
        return Err(format!("unknown options: {}", pyr_list(&unknown)));
    }
    if o.get("think").is_some_and(|v| !v.is_boolean()) || o.get("return_reasoning").is_some_and(|v| !v.is_boolean()) {
        return Err("think and return_reasoning must be booleans".into());
    }
    if o.get("max_think").is_some_and(|v| v.as_u64().is_none()) {
        return Err("max_think must be a non-negative integer".into());
    }
    if o.get("nothink_threshold").is_some_and(|v| !v.is_null() && !v.as_f64().is_some_and(|t| (0.0..=1.0).contains(&t))) {
        return Err("nothink_threshold must be null or a number in [0, 1]".into());
    }
    Ok(())
}

/// Check one question and its criteria.
fn check_question(qid: &str, raw: &Value) -> Result<(), String> {
    let o = raw.as_object().ok_or(format!("question {} must be an object", pyr(qid)))?;
    let kind = o.get("type").and_then(Value::as_str).unwrap_or("");
    if !["choice", "noul", "score"].contains(&kind) {
        return Err(format!("question {}: type must be one of ['choice', 'noul', 'score']", pyr(qid)));
    }
    let mut unknown: Vec<&str> = o.keys().map(String::as_str).filter(|k| !["type", "instructions", "criteria"].contains(k)).collect();
    if !unknown.is_empty() {
        unknown.sort();
        return Err(format!("question {}: unknown fields {}", pyr(qid), pyr_list(&unknown)));
    }
    let crit = o.get("criteria").unwrap_or(&Value::Null);
    match kind {
        "choice" if !crit.as_object().is_some_and(|c| (1..=MAX_OPTIONS).contains(&c.len())) => {
            Err(format!("question {}: choice criteria must be an object with 1..{MAX_OPTIONS} options", pyr(qid)))
        }
        "noul" if !crit.is_null() && !crit.as_object().is_some_and(|c| c.keys().all(|k| k == "true" || k == "false")) => {
            Err(format!("question {}: noul criteria may only describe true and false", pyr(qid)))
        }
        "score" if !crit.as_array().is_some_and(|c| (1..=MAX_OPTIONS).contains(&c.len())) => {
            Err(format!("question {}: score criteria must be a list of 1..{MAX_OPTIONS} levels", pyr(qid)))
        }
        _ => Ok(()),
    }
}

/// Answer fields of one question. Our option order: noul [true, false], choice in criteria order, score by level.
fn answer(r: &Request, q: &Value, p: &[f32]) -> Value {
    let p: Vec<f64> = p.iter().map(|&x| x as f64).collect();
    let k = p.len();
    match r.qtype.as_str() {
        "noul" => json!({"type": "noul", "noul": r2(p[0])}),
        "choice" => {
            let conf = if k == 1 { 1.0 } else { (p[argmax(&p)] - 1.0 / k as f64) / (1.0 - 1.0 / k as f64) };
            let probs: Map<String, Value> = r.options.iter().zip(&p).map(|(o, &x)| (o.id.clone(), json!(r2(x)))).collect();
            json!({"type": "choice", "choice": r.options[argmax(&p)].id, "confidence": r2(conf), "probabilities": probs})
        }
        _ => {
            let mode = argmax(&p);
            let conf = if k == 1 { 1.0 } else { 1.0 - p.iter().enumerate().map(|(i, x)| x * (i as f64 - mode as f64).abs()).sum::<f64>() / (k - 1) as f64 };
            let legend: Map<String, Value> = q["criteria"].as_array().unwrap().iter().enumerate()
                .map(|(i, v)| (i.to_string(), json!(match v { Value::String(s) => s.clone(), other => py_json(other) })))
                .collect();
            let probs: Map<String, Value> = p.iter().enumerate().map(|(i, &x)| (i.to_string(), json!(r2(x)))).collect();
            json!({"type": "score", "score": r2(p.iter().enumerate().map(|(i, x)| i as f64 * x).sum()), "legend": legend,
                   "probabilities": probs, "confidence": r2(conf)})
        }
    }
}

enum Reply {
    Ok(Value),
    Invalid(String),
}

fn systemone(m: &mut Model, name: &str, body: &str) -> Result<Reply> {
    let body: Value = match serde_json::from_str(if body.trim().is_empty() { "null" } else { body }) {
        Ok(v) => v,
        Err(e) => return Ok(Reply::Invalid(format!("invalid JSON: {e}"))),
    };
    let Some(b) = body.as_object() else { return Ok(Reply::Invalid("request body must be a JSON object".into())) };
    let Some(state) = b.get("state") else { return Ok(Reply::Invalid("state is required".into())) };
    let Some(qs) = b.get("questions").and_then(Value::as_object).filter(|q| !q.is_empty()) else {
        return Ok(Reply::Invalid("questions must be a non-empty object".into()));
    };
    let model = match b.get("model") {
        None => name.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => return Ok(Reply::Invalid("model must be a string".into())),
    };
    for (qid, q) in qs {
        if let Err(e) = check_question(qid, q) {
            return Ok(Reply::Invalid(e));
        }
    }
    if let Err(e) = check_options(b.get("options").unwrap_or(&Value::Null)) {
        return Ok(Reply::Invalid(e));
    }
    let questions: Vec<&Value> = qs.values().collect();
    let t0 = Instant::now();
    let res = m.decide_many(state, &questions)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let answers: Map<String, Value> = qs.iter().zip(&res).map(|((qid, q), (r, p, _))| (qid.clone(), answer(r, q, p))).collect();
    let answers = Value::Object(answers);
    Ok(Reply::Ok(json!({
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": res.iter().map(|x| x.2).sum::<usize>(), "output_tokens": 0, "reasoning_tokens": 0},
        "latency_ms": (ms * 10.0).round() / 10.0,
    })))
}

pub fn main(a: &[String]) -> Result<()> {
    let args = parse_args(a)?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads = if args.threads > cores {
        eprintln!("warning: --threads {} but only {cores} cores available; using {cores}", args.threads);
        cores
    } else {
        args.threads
    };
    let dir = Path::new(&args.dir);
    let name = dir.canonicalize()?.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let t0 = Instant::now();
    let mut m = Model::load(dir, threads)?;
    // one warm-up decision, so the first request is not the slow one
    m.decide_many(&json!("warmup"), &[&json!({"type": "noul", "instructions": "Is this a warmup?"})])?;
    let load_s = t0.elapsed().as_secs_f64();
    let addr = format!("{}:{}", args.host, args.port);
    let server = Server::http(&addr).map_err(|e| anyhow!("cannot listen on {addr}: {e}"))?;
    let info = json!({"model": name, "threads": threads, "load_s": (load_s * 100.0).round() / 100.0, "rss_mb": rss_mb().round()});
    println!("{}", json!({"serving": format!("http://{addr}"), "model": name, "threads": threads,
                          "load_s": info["load_s"], "rss_mb": info["rss_mb"]}));
    let ct = Header::from_bytes("content-type", "application/json").unwrap();
    let mut counter: u64 = 0;
    for mut req in server.incoming_requests() {
        let path = req.url().split('?').next().unwrap_or("").trim_end_matches('/').to_string();
        let (code, payload) = match (req.method(), path.as_str()) {
            (Method::Get, "/v1/models") => (200, json!({"models": [{"name": name, "description":
                "Jev at home: typed decisions (noul/choice/score) with calibrated probabilities, one forward pass per question, on a CPU.",
                "runtime": "jevhome", "threads": threads, "load_s": info["load_s"]}]})),
            (Method::Post, "/v1/systemone") => {
                let mut body = String::new();
                match req.as_reader().read_to_string(&mut body) {
                    Err(e) => (422, json!({"detail": format!("cannot read body: {e}")})),
                    Ok(_) => match systemone(&mut m, &name, &body) {
                        Ok(Reply::Ok(v)) => (200, v),
                        Ok(Reply::Invalid(e)) => (422, json!({"detail": e})),
                        Err(e) => (500, json!({"detail": e.to_string()})),
                    },
                }
            }
            _ => (404, json!({"detail": "not found"})),
        };
        counter += 1;
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let id = Header::from_bytes("x-typesafe-request-id", format!("{:016x}{:016x}", nanos as u64, counter)).unwrap();
        let resp = Response::from_string(payload.to_string()).with_status_code(code).with_header(ct.clone()).with_header(id);
        let _ = req.respond(resp);
    }
    Ok(())
}
