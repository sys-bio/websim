//! A front end for a basic subset of the Antimony modelling language.
//!
//! Supported:
//!
//! ```text
//! model example()            // optional `model ... end` wrapper
//!   J1: $Xo -> 2 S1; k1*Xo   // reaction: name, stoichiometry, '$' boundary species, rate law
//!   S1 + S2 => ; k2*S1*S2    // unnamed reaction; empty side; '->' and '=>' both work
//!   v := Vm*S1/(Km + S1)     // rule, recomputed at every step
//!   x' = -k*x                // rate rule
//!   k1 = 0.1; S1 = 0         // parameters and initial values; ';' separates statements
//!   const Xo                 // boundary species declared without '$'
//!   E1: at (time > 10 && S1 < 2): k1 = k1/2, S1 = 0   // event
//! end
//! ```
//!
//! Not supported (yet): event delays (`after`) and options (`t0=`, `priority=`,
//! `persistent=`, `fromTrigger=`), functions, compartments and units, initial
//! assignments to species, and `/* */` comments.
//!
//! Reactions are translated into plain statements: each reaction's rate law
//! becomes a rule, and each floating species gets a rate equation summing
//! stoichiometry × rate over the reactions it takes part in.

use crate::model::{EventStmt, ParseError, Stmt, StmtKind, is_identifier};

/// The statements and events in a model's text.
pub struct Parsed {
    pub stmts: Vec<Stmt>,
    pub events: Vec<EventStmt>,
    /// For each floating species changed by reactions, in order of first
    /// appearance: (species, [(reaction, net stoichiometry)]). A species that
    /// takes part only as a catalyst has an empty list. Used for conservation
    /// analysis; the rate statements in `stmts` carry the same information
    /// for simulation.
    pub stoichiometry: Vec<(String, Vec<(String, f64)>)>,
}

struct Reaction {
    name: String,
    line: usize,
    /// (stoichiometry, species): reactants with negative, products with positive stoichiometry.
    terms: Vec<(f64, String)>,
}

pub fn statements(text: &str) -> Result<Parsed, ParseError> {
    let mut stmts = Vec::new();
    let mut events = Vec::new();
    let mut reactions: Vec<Reaction> = Vec::new();
    let mut boundary: Vec<String> = Vec::new();

    let text = blank_block_comments(text);
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let err = |message: String| ParseError { line, message };
        let code = strip_comment(raw).trim();
        if code.is_empty() || code == "end" || code.starts_with("model ") || code.starts_with("model*") {
            continue;
        }

        let mut parts = code.split(';').map(str::trim);
        while let Some(part) = parts.next() {
            if part.is_empty() {
                continue;
            }
            if let Some(event) = parse_event(part, line).map_err(err)? {
                events.push(event);
            } else if part.contains("->") || part.contains("=>") {
                // A reaction's rate law is the next ';'-separated part.
                let rate_law = parts.next().filter(|r| !r.is_empty()).ok_or_else(|| {
                    err("a reaction needs a rate law after ';', e.g.  S1 -> S2; k1*S1".to_owned())
                })?;
                let reaction = parse_reaction(part, reactions.len(), line, &mut boundary).map_err(err)?;
                stmts.push(Stmt {
                    line,
                    name: reaction.name.clone(),
                    kind: StmtKind::Rule,
                    rhs: rate_law.to_owned(),
                });
                reactions.push(reaction);
            } else {
                parse_statement(part, line, &mut stmts, &mut boundary).map_err(err)?;
            }
        }
    }

    // Each floating species changes by Σ stoichiometry × rate over its reactions.
    // (species, line of first reaction, net stoichiometry per reaction)
    let mut species: Vec<(String, usize, Vec<(String, f64)>)> = Vec::new();
    for r in &reactions {
        for (stoich, name) in &r.terms {
            if boundary.contains(name) {
                continue;
            }
            let i = match species.iter().position(|(s, ..)| s == name) {
                Some(i) => i,
                None => {
                    species.push((name.clone(), r.line, Vec::new()));
                    species.len() - 1
                }
            };
            let changes = &mut species[i].2;
            match changes.iter_mut().find(|(j, _)| *j == r.name) {
                Some((_, net)) => *net += stoich,
                None => changes.push((r.name.clone(), *stoich)),
            }
        }
    }

    let mut stoichiometry = Vec::new();
    for (name, line, changes) in species {
        if let Some(rate_rule) = stmts.iter().find(|s| s.kind == StmtKind::Rate && s.name == name) {
            return Err(ParseError {
                line: rate_rule.line,
                message: format!(
                    "'{name}' is changed by reactions, so it can't also have a rate rule \
                     (mark it '$' in the reactions to make it a boundary species)"
                ),
            });
        }
        let terms: Vec<String> = changes
            .iter()
            .filter(|(_, net)| *net != 0.0)
            .map(|(reaction, net)| format!("({net})*{reaction}"))
            .collect();
        let rhs = if terms.is_empty() { "0".to_owned() } else { terms.join(" + ") };
        stmts.push(Stmt { line, name: name.clone(), kind: StmtKind::Rate, rhs });
        let net: Vec<(String, f64)> = changes.into_iter().filter(|(_, n)| *n != 0.0).collect();
        stoichiometry.push((name, net));
    }

    // Like Antimony, boundary species without a value start at 0.
    for name in boundary {
        if !stmts.iter().any(|s| s.name == name) {
            stmts.push(Stmt { line: 0, name, kind: StmtKind::Assign, rhs: "0".to_owned() });
        }
    }
    Ok(Parsed { stmts, events, stoichiometry })
}

/// Replace the contents of every `/* ... */` comment with spaces, keeping the
/// line breaks so that line numbers in error messages stay right.
fn blank_block_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_comment = false;
    while let Some(c) = chars.next() {
        if in_comment {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                out.push_str("  ");
                in_comment = false;
            } else {
                out.push(if c == '\n' { '\n' } else { ' ' });
            }
        } else if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            out.push_str("  ");
            in_comment = true;
        } else if (c == '/' && chars.peek() == Some(&'/')) || c == '#' {
            // A line comment: copy it unchanged, so a '/*' inside it starts nothing.
            out.push(c);
            while let Some(&next) = chars.peek() {
                if next == '\n' {
                    break;
                }
                out.push(next);
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// If `text` is an event, `[name:] at trigger: target = value, ...`, parse it.
fn parse_event(text: &str, line: usize) -> Result<Option<EventStmt>, String> {
    fn starts_with_at(s: &str) -> bool {
        s.strip_prefix("at")
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_whitespace() || c == '('))
    }
    let (name, rest) = match text.split_once(':') {
        Some((name, rest)) if is_identifier(name.trim()) && starts_with_at(rest.trim_start()) => {
            (Some(name.trim().to_owned()), rest.trim_start())
        }
        _ => (None, text),
    };
    if !starts_with_at(rest) {
        return Ok(None);
    }
    let rest = rest["at".len()..].trim();

    // The trigger runs up to the first ':' outside parentheses.
    let colon = find_top_level(rest, ':').ok_or(
        "an event needs ':' between its condition and assignments, e.g.  at (time > 10): k1 = 2",
    )?;
    let (trigger, assignments) = (rest[..colon].trim(), rest[colon + 1..].trim());
    if find_top_level(trigger, ',').is_some() {
        return Err("event options (t0=, priority=, persistent=, fromTrigger=) aren't supported yet".to_owned());
    }
    if trigger.split(|c: char| !(c.is_alphanumeric() || c == '_')).any(|word| word == "after") {
        return Err("event delays ('after') aren't supported yet".to_owned());
    }
    if trigger.is_empty() {
        return Err("an event needs a condition after 'at'".to_owned());
    }

    let mut parsed = Vec::new();
    for assignment in split_top_level(assignments, ',') {
        // Split at the first '=' that isn't part of '==', '<=', '>=' or '!='.
        let bytes = assignment.as_bytes();
        let equals = (0..bytes.len()).find(|&i| {
            bytes[i] == b'='
                && bytes.get(i + 1) != Some(&b'=')
                && (i == 0 || !b"=<>!".contains(&bytes[i - 1]))
        });
        let Some(equals) = equals else {
            return Err(format!("expected `name = value` in the event, not '{}'", assignment.trim()));
        };
        let target = assignment[..equals].trim();
        if !is_identifier(target) {
            return Err(format!("'{target}' is not a valid name to set in an event"));
        }
        parsed.push((target.to_owned(), assignment[equals + 1..].trim().to_owned()));
    }
    if parsed.is_empty() {
        return Err("an event needs at least one assignment after ':'".to_owned());
    }
    Ok(Some(EventStmt {
        line,
        name,
        trigger: trigger.to_owned(),
        assignments: parsed,
    }))
}

/// The position of the first `ch` in `text` that isn't inside parentheses.
fn find_top_level(text: &str, ch: char) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            c if c == ch && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Split `text` at each `ch` that isn't inside parentheses, so that commas
/// inside function calls like max(a, b) don't split. Empty pieces are dropped.
fn split_top_level(text: &str, ch: char) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while let Some(i) = find_top_level(rest, ch) {
        pieces.push(&rest[..i]);
        rest = &rest[i + ch.len_utf8()..];
    }
    pieces.push(rest);
    pieces.into_iter().filter(|p| !p.trim().is_empty()).collect()
}

/// Everything before a `#` or `//` comment.
fn strip_comment(line: &str) -> &str {
    let end = [line.find('#'), line.find("//")]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(line.len());
    &line[..end]
}

/// Parse `name: A + 2 B -> C` (the part before the rate law).
fn parse_reaction(
    text: &str,
    index: usize,
    line: usize,
    boundary: &mut Vec<String>,
) -> Result<Reaction, String> {
    let arrow = text.find("->").or_else(|| text.find("=>")).unwrap();
    let (head, products) = (&text[..arrow], &text[arrow + 2..]);

    let (name, reactants) = match head.split_once(':') {
        Some((name, reactants)) => {
            let name = name.trim();
            if !is_identifier(name) {
                return Err(format!("'{name}' is not a valid reaction name"));
            }
            (name.to_owned(), reactants)
        }
        // Unnamed reactions get a name the user can't type, so it can't clash.
        None => (format!("_J{index}"), head),
    };

    let mut terms = Vec::new();
    for (stoich, species) in parse_side(reactants, boundary)? {
        terms.push((-stoich, species));
    }
    terms.extend(parse_side(products, boundary)?);
    Ok(Reaction { name, line, terms })
}

/// Parse one side of a reaction, e.g. `2 A + $B`. An empty side is allowed.
fn parse_side(side: &str, boundary: &mut Vec<String>) -> Result<Vec<(f64, String)>, String> {
    let side = side.trim();
    if side.is_empty() {
        return Ok(Vec::new());
    }
    side.split('+')
        .map(|term| {
            let term = term.trim();
            // Optional stoichiometry: `2 A` or `2A`.
            let digits_end = term
                .find(|c: char| !(c.is_ascii_digit() || c == '.'))
                .unwrap_or(term.len());
            let (number, rest) = term.split_at(digits_end);
            let stoich = if number.is_empty() {
                1.0
            } else {
                number.parse().map_err(|_| format!("'{number}' is not a valid stoichiometry"))?
            };
            let rest = rest.trim();
            let name = match rest.strip_prefix('$') {
                Some(name) => {
                    mark_boundary(name.trim(), boundary);
                    name.trim()
                }
                None => rest,
            };
            if !is_identifier(name) {
                return Err(format!("'{term}' is not a valid species in a reaction"));
            }
            Ok((stoich, name.to_owned()))
        })
        .collect()
}

fn mark_boundary(name: &str, boundary: &mut Vec<String>) {
    if !boundary.iter().any(|b| b == name) {
        boundary.push(name.to_owned());
    }
}

/// Parse a statement that isn't a reaction: an assignment, rule, rate rule or declaration.
fn parse_statement(
    text: &str,
    line: usize,
    stmts: &mut Vec<Stmt>,
    boundary: &mut Vec<String>,
) -> Result<(), String> {
    // Leading keywords: `const` marks a boundary species; the others are just declarations.
    let mut text = text;
    let mut is_const = false;
    loop {
        let (word, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        match word {
            "const" => is_const = true,
            "var" | "species" | "compartment" => {}
            "function" | "unit" | "import" | "delete" => {
                return Err(format!("'{word}' isn't supported by this basic Antimony parser"));
            }
            _ => break,
        }
        text = rest.trim();
    }
    let make = |name: &str, kind: StmtKind, rhs: &str| Stmt {
        line,
        name: name.trim().to_owned(),
        kind,
        rhs: rhs.trim().to_owned(),
    };

    if let Some((lhs, rhs)) = text.split_once(":=") {
        stmts.push(make(lhs, StmtKind::Rule, rhs));
    } else if let Some((lhs, rhs)) = text.split_once('=') {
        let lhs = lhs.trim();
        if let Some(name) = lhs.strip_suffix('\'') {
            stmts.push(make(name, StmtKind::Rate, rhs));
        } else {
            let name = match lhs.strip_prefix('$') {
                Some(name) => {
                    mark_boundary(name.trim(), boundary);
                    name
                }
                None => lhs,
            };
            if is_const {
                mark_boundary(name.trim(), boundary);
            }
            stmts.push(make(name, StmtKind::Assign, rhs));
        }
    } else {
        // A bare declaration like `const Xo, $X1` or `species S1 in cell`.
        for name in text.split(',') {
            let name = name.split_whitespace().next().unwrap_or("");
            match name.strip_prefix('$') {
                Some(name) => mark_boundary(name, boundary),
                None if is_const => mark_boundary(name, boundary),
                None => {}
            }
            if !is_identifier(name.trim_start_matches('$')) {
                return Err(format!("don't understand '{text}'"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::model::Model;
    use crate::solvers::SolverSettings;

    fn species_names(model: &Model) -> Vec<&str> {
        model.species.iter().map(|s| s.name.as_str()).collect()
    }

    #[test]
    fn reactions_with_stoichiometry_conserve_mass() {
        // 2A + B -> C: A + 2C stays constant, and so does B + C.
        let model = Model::parse("J1: 2 A + B -> C; k*A*B\nk = 0.1; A = 10; B = 4").unwrap();
        assert_eq!(species_names(&model), ["A", "B", "C"]);
        let end = model.simulate(5.0, &SolverSettings::rk4(1000)).y.last().unwrap().clone();
        assert!(end[2] > 0.1, "some C was made");
        assert!((end[0] + 2.0 * end[2] - 10.0).abs() < 1e-9);
        assert!((end[1] + end[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn boundary_species_are_constant() {
        let model = Model::parse("$X -> S; k*X\nS -> ; S\nX = 2; k = 1").unwrap();
        assert_eq!(species_names(&model), ["S"]);
        let end = model.simulate(30.0, &SolverSettings::rk4(3000)).y.last().unwrap().clone();
        assert!((end[0] - 2.0).abs() < 1e-6, "S = {}", end[0]);

        // `const` works the same as '$'.
        let model = Model::parse("const X\nX -> S; k*X\nS -> ; S\nX = 2; k = 1").unwrap();
        assert_eq!(species_names(&model), ["S"]);
    }

    #[test]
    fn catalysts_have_no_net_change() {
        let model = Model::parse("E + S -> E + P; k*E*S\nE = 1; S = 5; k = 1").unwrap();
        let end = model.simulate(1.0, &SolverSettings::rk4(1000)).y.last().unwrap().clone();
        assert_eq!(end[0], 1.0); // E is unchanged
    }

    #[test]
    fn model_wrapper_comments_and_semicolons() {
        let text = "model test()\n  J1: S1 -> S2; k1*S1; k1 = 0.5; // comment\n  S1 = 3 # comment\nend";
        let model = Model::parse(text).unwrap();
        assert_eq!(species_names(&model), ["S1", "S2"]);
        assert_eq!(model.species[0].value, 3.0);
    }

    #[test]
    fn helpful_errors() {
        let err = |text| Model::parse(text).unwrap_err().to_string();
        assert!(err("S1 -> S2").contains("needs a rate law"));
        assert!(err("S1 -> S2; k*S1\nS1' = 1\nk = 1").contains("can't also have a rate rule"));
        assert!(err("function f(x)").contains("isn't supported"));
        assert!(err("J1: S1 -> 2x Y; 1").contains("not a valid species"));
    }
}
