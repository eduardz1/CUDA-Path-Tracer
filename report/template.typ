#import "@preview/codly:1.1.1": *
#import "@preview/codly-languages:0.1.3": *

#let balance(content) = layout(size => {
  let count = content.at("count")
  let textheight = measure(content).at("height")
  let linegap = par.leading.em * textheight
  let (height,) = measure(block(width: size.width, content))
  let lines = calc.ceil((height - textheight) / count / (textheight + linegap))
  let newheight = lines * (textheight + linegap) + textheight
  [#block(height: newheight)[#content]]
})

#let template(
  title: [],
  authors: (),
  lang: "en",
  body,
) = {
  set document(title: title, author: authors)
  set text(font: "Minion Pro", size: 10pt, lang: lang, fallback: false)
  set page(paper: "a4", numbering: "1")
  set par(justify: true)

  show figure.caption: emph

  show heading: smallcaps

  {
    // Title Page
    set align(center + horizon)

    v(0.5fr)

    image("imgs/logo.png", width: 60%)
    line(length: 60%)
    block(
      smallcaps(
        text(
          size: 2.8em,
          weight: "bold",
          title,
        ),
      ),
    )
    line(length: 60%)

    // Author and Academic Year
    block()[
      #authors.map(author => {
        text(author)
      }).join(", ")
    ]

    v(1fr)
  }
  pagebreak()

  set outline(fill: repeat[ #sym.space ⋅ ], indent: true)
  show outline.entry.where(level: 1): it => {
    v(1.2em, weak: true)
    strong(it)
  }

  show raw: set text(ligatures: true, font: "Fira Code", fallback: false)
  show: codly-init
  codly(
    languages: codly-languages,
    zebra-fill: none,
    number-format: it => text(fill: luma(200), str(it)),
  )

  body
}