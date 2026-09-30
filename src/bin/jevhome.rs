//! jevhome serve <model_dir> [--port 8009] [--host 127.0.0.1] [--threads 4]   HTTP API (POST /v1/systemone, GET /v1/models)
//! jevhome probe <model_dir> <items.jsonl> <out.jsonl> [threads]   one decision per item, one at a time
//! jevhome bench <model_dir> <items.jsonl> <out_dir> [threads]    latency benchmark (single decisions only)
//! jevhome statetext <items.jsonl>                                 state_text() of every row (JSON string per line)
use std::io::{BufRead, Write};
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Value};


/// ONNX Runtime is linked into the binary; with the `dynamic` feature it is loaded from ORT_DYLIB_PATH instead.
fn init_ort() -> Result<()> {
    #[cfg(feature = "dynamic")]
    {
        let lib = std::env::var("ORT_DYLIB_PATH").expect("set ORT_DYLIB_PATH to libonnxruntime.so");
        ort::init_from(&lib)?.commit();
    }
    Ok(())
}

fn status_mb(key: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// Median as Python's statistics.median, other percentiles by nearest rank.
fn pct(xs: &[f64], q: f64) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if q == 50.0 {
        return if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 };
    }
    v[((q / 100.0 * (n - 1) as f64).round() as usize).min(n - 1)]
}

fn rows(path: &str) -> Result<Vec<Value>> {
    let f = std::io::BufReader::new(std::fs::File::open(path)?);
    Ok(f.lines().filter_map(|l| l.ok()).filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(&l)).collect::<Result<_, _>>()?)
}

fn main() -> Result<()> {
    let t0 = Instant::now();
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("serve") => {
            init_ort()?;
            jevhome::serve::main(&a[2..])?;
        }
        Some("statetext") => {
            let out = std::io::stdout();
            let mut out = out.lock();
            for r in rows(&a[2])? {
                writeln!(out, "{}", serde_json::to_string(&jevhome::state_text(&r["state"]))?)?;
            }
        }
        Some("probe") => {
            init_ort()?;
            let threads: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(4);
            let rss_start = status_mb("VmRSS:");
            let mut m = jevhome::Model::load(std::path::Path::new(&a[2]), threads)?;
            let ready_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss_loaded = status_mb("VmRSS:");
            if a[3] == "-" {
                // streaming mode (items path "-"): one task per stdin line, one answer per
                // stdout line, flushed at once; the load statistics go to stderr
                let mut stdout = std::io::stdout().lock();
                for line in std::io::stdin().lock().lines() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let r: Value = serde_json::from_str(&line)?;
                    let (req, p, enc) = m.decide(&r)?;
                    let ids: Vec<&str> = req.options.iter().map(|o| o.id.as_str()).collect();
                    writeln!(stdout, "{}", json!({"id": r["id"], "options": ids, "probs": p,
                                                  "input_tokens": enc.ids.len() + enc.query_ids.iter().map(Vec::len).sum::<usize>()}))?;
                    stdout.flush()?;
                }
                eprintln!("{}", json!({"ready_ms": ready_ms, "rss_start_mb": rss_start, "rss_after_load_mb": rss_loaded,
                                       "peak_rss_mb": status_mb("VmHWM:")}));
                return Ok(());
            }
            let mut out = std::fs::File::create(&a[4])?;
            for r in rows(&a[3])? {
                let (req, p, enc) = m.decide(&r)?;
                let ids: Vec<&str> = req.options.iter().map(|o| o.id.as_str()).collect();
                writeln!(out, "{}", json!({"id": r["id"], "options": ids, "probs": p, "ids": enc.ids, "query_ids": enc.query_ids}))?;
            }
            println!("{}", json!({"ready_ms": ready_ms, "rss_start_mb": rss_start, "rss_after_load_mb": rss_loaded,
                                  "rss_end_mb": status_mb("VmRSS:"), "peak_rss_mb": status_mb("VmHWM:")}));
        }
        Some("bench") => {
            // benchmark protocol: load, 1 cold decision, 10 warm-ups (items[(i*7)%n]), then every item
            // once, timed end to end (tokenisation, request building, ONNX run(s), calibrated probabilities).
            // No batch-8 pass: the Rust runtime decides one item at a time.
            init_ort()?;
            let dir = std::path::Path::new(&a[2]);
            let threads: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(4);
            let rss_pre = status_mb("VmRSS:");
            let tl = Instant::now();
            let mut m = jevhome::Model::load(dir, threads)?;
            let load_s = tl.elapsed().as_secs_f64();
            let items = rows(&a[3])?;
            let tc = Instant::now();
            m.decide(&items[0])?;
            let cold_ms = tc.elapsed().as_secs_f64() * 1000.0;
            for i in 0..10 {
                m.decide(&items[(i * 7) % items.len()])?;
            }
            let out_dir = std::path::Path::new(&a[4]);
            std::fs::create_dir_all(out_dir)?;
            let mut fh = std::fs::File::create(out_dir.join("items.jsonl"))?;
            let (mut e2e, mut toks) = (Vec::new(), Vec::new());
            for r in &items {
                let t = Instant::now();
                let (_, p, enc) = m.decide(r)?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let n = enc.ids.len() + enc.query_ids.iter().map(Vec::len).sum::<usize>();
                writeln!(fh, "{}", json!({"set": r["_set"], "id": r["id"], "tokens": n, "e2e_ms": ms, "probs": p}))?;
                e2e.push(ms);
                toks.push(n as f64);
            }
            let disk: u64 = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    !n.contains(".opt.") && !n.ends_with(".json")
                })
                .map(|e| e.metadata().map(|md| md.len()).unwrap_or(0))
                .sum();
            let peak = status_mb("VmHWM:");
            let summ = json!({"model": a.get(6).cloned().unwrap_or_default(), "checkpoint": dir.file_name().unwrap().to_string_lossy(),
                "runtime": "rust-ort", "threads": threads, "load_s": load_s, "cold_first_ms": cold_ms, "n": e2e.len(),
                "tokens_p50": pct(&toks, 50.0), "tokens_p90": pct(&toks, 90.0),
                "e2e_ms": {"mean": e2e.iter().sum::<f64>() / e2e.len() as f64, "p50": pct(&e2e, 50.0), "p90": pct(&e2e, 90.0)},
                "peak_rss_mb": peak, "rss_before_load_mb": rss_pre, "model_rss_mb": peak - rss_pre,
                "disk_mb": disk as f64 / 1048576.0});
            std::fs::write(out_dir.join("summary.json"), serde_json::to_string_pretty(&summ)?)?;
            println!("{summ}");
        }
        _ => eprintln!("usage:\n  jevhome serve <model_dir> [--port 8009] [--host 127.0.0.1] [--threads 4]\n  jevhome probe <model_dir> <items.jsonl> <out.jsonl> [threads]\n  jevhome bench <model_dir> <items.jsonl> <out_dir> [threads] [name]\n  jevhome statetext <items.jsonl>"),
    }
    Ok(())
}
