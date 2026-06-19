use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod fast_chunker {
    use pyo3::prelude::*;

    /// Slices a massive string into smaller overlapping character blocks safely.
    #[pyfunction]
    fn chunk_text(text: String, chunk_size: usize, chunk_overlap: usize) -> PyResult<Vec<String>> {
        // Safety Guard: Prevent infinite loops or negative math errors
        if chunk_overlap >= chunk_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_overlap must be strictly smaller than chunk_size"
            ));
        }
    
        // Safe character extraction
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        // High-Speed sliding window loop
        while start < chars.len() {
            // Guarantee code does not look past the final characters of the text
            let end = std::cmp::min(start + chunk_size, chars.len());

            // Take the slice of characters and assemble them back into a clean String
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);

            // Break early if code reaches the absolute end of the book text
            if end == chars.len() {
                break;
            }

            // Move the window forward by the stride distance
            start += chunk_size - chunk_overlap;
        }

        // Return the compile list of chunks to Python
        Ok(chunks)
    }
}
