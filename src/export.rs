//! Saving a text file: a Save dialog on the desktop, a download in the browser.

/// Ask where to save `contents`, suggesting `file_name`, and write it there.
/// Returns `Ok(Some(where))` when saved, `Ok(None)` if the user cancelled.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_text_file(file_name: &str, contents: &str) -> Result<Option<String>, String> {
    let Some(path) = rfd::FileDialog::new()
        .set_file_name(file_name)
        .add_filter("CSV files", &["csv"])
        .save_file()
    else {
        return Ok(None);
    };
    std::fs::write(&path, contents).map_err(|e| format!("Couldn't save {}: {e}", path.display()))?;
    Ok(Some(path.display().to_string()))
}

/// A web page can't write files, so offer `contents` as a download named `file_name`:
/// put it in a Blob, link to it, and click the link.
#[cfg(target_arch = "wasm32")]
pub fn save_text_file(file_name: &str, contents: &str) -> Result<Option<String>, String> {
    use eframe::wasm_bindgen::{JsCast as _, JsValue};

    let js_error = |e: JsValue| format!("Couldn't download the file: {e:?}");

    let parts = js_sys::Array::of1(&JsValue::from_str(contents));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("text/csv");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).map_err(js_error)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(js_error)?;

    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or("Couldn't download the file: no document")?;
    let link: web_sys::HtmlAnchorElement = document
        .create_element("a")
        .map_err(js_error)?
        .dyn_into()
        .map_err(|_| "Couldn't download the file: not a link element".to_owned())?;
    link.set_href(&url);
    link.set_download(file_name);
    link.click();
    web_sys::Url::revoke_object_url(&url).map_err(js_error)?;

    Ok(Some(format!("{file_name} (in your downloads)")))
}
