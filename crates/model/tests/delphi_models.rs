//! Checks against the Delphi Bifurcata project's bundled models and baselines,
//! read in place (nothing there is changed). Skipped when that project is not
//! on this machine; set BIFURCATA_DELPHI_DIR to point elsewhere.
//!
//! * Every bundled `.ant` model must load.
//! * For each `ant_*.json` baseline, our steady state at the run's starting
//!   parameter value must match the baseline's first point — which libRoadRunner
//!   found — species by species, by name.

use std::path::PathBuf;

use websim_model::model::Model;
use websim_model::steady::find_steady_state;

fn delphi_dir() -> Option<PathBuf> {
    let dir = std::env::var("BIFURCATA_DELPHI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"D:\Documents\Embarcadero\Studio\Projects\Bifurcation_Delphi"));
    if dir.join("GUIApp").join("models").is_dir() {
        Some(dir)
    } else {
        eprintln!("skipped: the Delphi Bifurcata project is not at {}", dir.display());
        None
    }
}

#[test]
fn every_bundled_model_loads() {
    let Some(dir) = delphi_dir() else { return };
    let mut failures = Vec::new();
    let mut count = 0;
    for entry in std::fs::read_dir(dir.join("GUIApp").join("models")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "ant") {
            count += 1;
            let text = std::fs::read_to_string(&path).unwrap();
            if let Err(e) = Model::parse(&text) {
                failures.push(format!("{}: {e}", path.file_name().unwrap().to_string_lossy()));
            }
        }
    }
    assert!(count >= 20, "expected the bundled models, found {count}");
    assert!(failures.is_empty(), "models that failed to load:\n  {}", failures.join("\n  "));
}

#[test]
fn steady_states_match_the_baselines() {
    let Some(dir) = delphi_dir() else { return };
    let mut report = Vec::new();
    let mut mismatches = 0;
    let mut compared = 0;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir.join("baselines"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("ant_"))
        .collect();
    paths.sort();

    for path in paths {
        // Delphi writes a UTF-8 byte-order mark, which serde_json rejects.
        let json = std::fs::read_to_string(&path).unwrap();
        let baseline: serde_json::Value = serde_json::from_str(json.trim_start_matches('\u{feff}')).unwrap();
        let model_name = baseline["model"]["name"].as_str().unwrap();
        let text = std::fs::read_to_string(dir.join("GUIApp").join("models").join(format!("{model_name}.ant"))).unwrap();
        let model = Model::parse(&text).unwrap();

        let parameter = baseline["run"]["activeParameters"][0].as_str().unwrap();
        let point = &baseline["branches"][0]["points"][0];
        let lambda0 = point["lambda"][0].as_f64().unwrap();
        let names: Vec<&str> = baseline["model"]["stateNames"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let expected: Vec<f64> = point["u"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let mut params = model.parameter_values();
        let Some(k) = model.parameter_index(parameter) else {
            report.push(format!("{model_name}: parameter '{parameter}' not found"));
            mismatches += 1;
            continue;
        };
        params[k] = lambda0;

        // Runs that regen.bat starts by hand with --start (its `:antstart` calls).
        let file = path.file_stem().unwrap().to_string_lossy();
        let x0 = match file.as_ref() {
            "ant_pp2_trivial" | "ant_pp2_switch" => vec![0.0, 0.0],
            "ant_pp2_coexist" => vec![0.33333333333333, 2.0],
            _ => model.initial_state(),
        };

        match find_steady_state(&model, &x0, &params) {
            Err(e) => {
                mismatches += 1;
                report.push(format!("{model_name}: no steady state: {e}"));
            }
            Ok(ss) => {
                let mut worst = 0.0f64;
                let mut worst_name = "";
                for (name, want) in names.iter().zip(&expected) {
                    let i = model.species.iter().position(|s| s.name == *name).expect("species by name");
                    let err = (ss.state[i] - want).abs() / want.abs().max(1e-6);
                    if err > worst {
                        worst = err;
                        worst_name = name;
                    }
                }
                compared += 1;
                let ok = worst < 1e-5;
                if !ok {
                    mismatches += 1;
                }
                report.push(format!(
                    "{} {model_name}: {} species ({} independent, libRoadRunner {}), worst relative difference {worst:.1e} ({worst_name}){}",
                    if ok { "ok  " } else { "FAIL" },
                    model.species.len(),
                    ss.reduced.len(),
                    names.len(),
                    ss.warning.as_ref().map(|_| " [negative values]").unwrap_or("")
                ));
            }
        }
    }
    eprintln!("{}", report.join("\n"));
    assert!(compared >= 15, "expected the ant_* baselines, compared {compared}");
    assert_eq!(mismatches, 0, "steady states that differ from the baselines:\n{}", report.join("\n"));
}
