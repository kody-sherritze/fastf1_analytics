# Publishing Visuals

The published site is built from static files. A plot run produces the visual asset; the documentation generators turn its metadata into site content. MkDocs does not fetch FastF1 data while serving or building the site.

## Output pair

The plot scripts write two related files under `docs/assets/gallery/`: a PNG containing the rendered figure and a YAML sidecar containing metadata such as `title`, `subtitle`, `image`, `code_path`, `function`, `params`, and `tags`.

The sidecar is the handoff between analysis execution and documentation generation. Its `image` path is used in Markdown, while `code_path` and `params` make the source run discoverable.

## Gallery generation

`python tools/generate_gallery.py` reads all YAML files in `docs/assets/gallery/`, sorts them by filename, and rewrites the content between the `AUTO-GALLERY:BEGIN` and `AUTO-GALLERY:END` markers in `docs/gallery.md`. Content outside those markers is preserved.

The generated gallery includes the image, subtitle, source link, and compact parameter preview. Change the sidecar or generator rather than hand-editing the generated block.

## Case-study generation

`python tools/generate_case_studies.py` reads the same sidecars and applies them to `docs/case-studies/_template.md`. Case studies provide the narrative interpretation of a visual, while this section explains the underlying workflow. A sidecar can opt out of case-study generation with `case_study: false`.

## Build the site

After generating assets and pages, run:

```bash
mkdocs build
```

This checks that navigation, Markdown, links, plugins, and generated content can be assembled into the static site.
