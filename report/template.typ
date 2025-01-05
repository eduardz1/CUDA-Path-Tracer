#import "@preview/codly:1.1.1": *
#import "@preview/codly-languages:0.1.3": *
#import "@preview/lovelace:0.3.0": *


#let eqcolumns(n, gutter: 4%, content) = {
  layout(size => [
    #let (height,) = measure(
      block(
        width: (1 / n) * size.width * (1 - float(gutter) * n),
        content,
      ),
    )
    #block(
      height: height / n,
      columns(n, gutter: gutter, content),
    )
  ])
}

#let template(
  title: [],
  subtitle: [],
  authors: (),
  lang: "en",
  bibliography-file: "works.bib",
  body,
) = {
  set document(title: title, author: authors)
  set text(font: "New Computer Modern", size: 10pt, lang: lang, fallback: false)
  set page(paper: "a4")
  set par(justify: true, first-line-indent: 1.8em)

  show figure.caption: emph

  set heading(numbering: "1.1")
  show heading: smallcaps
  show heading: set block(above: 1.4em, below: 1em)
  set align(horizon)

  {
    // Title Page
    set align(center)
    set page(footer: text(fill: gray)[ #subtitle \ #datetime.today().display()])
    let width = 70%

    v(0.1fr)

    image("imgs/logo.png", width: width)
    line(length: width, stroke: 4pt)
    block(
      smallcaps(
        text(
          size: 3.2em,
          title,
        ),
      ),
    )
    line(length: width)

    // Author and Academic Year
    box(
      width: width,
      grid(columns: authors.len(), column-gutter: 1fr, ..authors),
    )

    v(1fr)
  }


  set outline(fill: repeat[ #sym.space #sym.dot.c ], indent: true)
  show outline.entry.where(level: 1): it => {
    v(1.2em, weak: true)
    strong(it)
  }

  show raw: set text(font: "Fira Code")
  show raw: set text(size: 0.8em)
  show: codly-init
  codly(
    languages: codly-languages,
    zebra-fill: none,
    number-format: it => text(fill: luma(200), str(it)),
  )

  outline()

  pagebreak()

  set page(numbering: "1")

  body

  pagebreak()
  bibliography(bibliography-file, full: true)
}

#let my-lovelace-defaults = (
  line-numbering: "1:",
  booktabs-stroke: 1pt + black,
)

#let pseudocode = pseudocode.with(..my-lovelace-defaults)
#let pseudocode-list = pseudocode-list.with(..my-lovelace-defaults)
