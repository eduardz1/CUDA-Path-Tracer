#let template(
  title: [],
  authors: (),
  lang: "en",
  body,
) = {
  set document(title: title, author: authors)
  set text(font: "Minion Pro", size: 10pt, lang: lang)
  set page(paper: "a4", numbering: "1")
  set par(justify: true)

  show figure.caption: emph

  show heading: smallcaps

  set outline(fill: repeat[ #sym.space ⋅ ], indent: true)
  show outline.entry.where(level: 1): it => {
    v(1.2em, weak: true)
    strong(it)
  }

  body
}