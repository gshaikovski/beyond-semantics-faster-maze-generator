//! The on-disk record format, at the byte level.
//!
//! Both `--exclude` and `--puzzle-file` read back files this tool wrote, and both parse them by
//! hand rather than through `serde_json`: only a slice of each line is ever needed, and a
//! multi-million-line scan cannot afford to decode the trace. The shared assumptions live here
//! so the two readers cannot drift apart.
//!
//! The format is a fixed key order and an escape-free `text` value (see `serializer`), so the
//! value ends at the next `"` — no unescaping, no state machine.

/// Read buffer for a line scan.
pub const READ_BUFFER: usize = 8 * 1024 * 1024;

/// Keys of the emitted JSON record.
pub const TEXT_KEY: &str = "\"text\":\"";
pub const ROWS_KEY: &str = "\"rows\":";
pub const COLS_KEY: &str = "\"cols\":";

/// Separator between the prompt (walls and query) and the search trace.
pub const TRACE_MARKER: &str = " reasoning";

/// Where a record's `text` value sits in its raw line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextSpan {
    /// First byte of the value, just past the opening quote.
    pub value_start: usize,
    /// The value's closing quote (an index *at* the `"`, so `line[close_quote..]` is the rest
    /// of the record, verbatim).
    pub close_quote: usize,
}

/// Locate the `text` value in a raw JSONL line.
pub fn locate_text(line: &str) -> Result<TextSpan, String> {
    let at = line
        .find(TEXT_KEY)
        .ok_or_else(|| format!("no {TEXT_KEY} key"))?;
    let value_start = at + TEXT_KEY.len();
    let end = line[value_start..]
        .find('"')
        .ok_or_else(|| "unterminated \"text\" value".to_string())?;
    Ok(TextSpan {
        value_start,
        close_quote: value_start + end,
    })
}

/// Read the integer value of `key` out of a slice of the record.
pub fn parse_field(region: &str, key: &str) -> Result<usize, String> {
    let at = region.find(key).ok_or_else(|| format!("no {key} key"))?;
    region[at + key.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .map_err(|_| format!("unparsable {key} value"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE: &str = r#"{"idx":7,"text":"query start 0 0 end","rows":3,"cols":3}"#;

    #[test]
    fn span_brackets_the_text_value() {
        let span = locate_text(LINE).unwrap();
        assert_eq!(
            &LINE[span.value_start..span.close_quote],
            "query start 0 0 end"
        );
        // The tail starts at the closing quote, so splicing keeps every later key verbatim.
        assert_eq!(&LINE[span.close_quote..], r#"","rows":3,"cols":3}"#);
    }

    #[test]
    fn missing_or_unterminated_values_are_errors() {
        assert!(locate_text(r#"{"idx":1}"#).unwrap_err().contains("text"));
        // Python's default json.dumps spacing (`"text": "`) does not match the key.
        assert!(locate_text(r#"{"text": "abc"}"#).is_err());
        assert!(locate_text(r#"{"text":"abc}"#)
            .unwrap_err()
            .contains("unterminated"));
    }

    #[test]
    fn fields_parse_from_any_region() {
        assert_eq!(parse_field(LINE, ROWS_KEY).unwrap(), 3);
        assert!(parse_field(LINE, "\"dim\":").is_err());
        assert!(parse_field(r#""rows":x"#, ROWS_KEY).is_err());
    }
}
