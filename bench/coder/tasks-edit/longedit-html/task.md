Make these changes to index.html. Leave everything else as it is.

1. Change the theme: `--accent` becomes #2c6fb5 and `--paper` becomes #eef3f7.
2. Rename the CSS class `card-note` to `margin-note` everywhere it appears (the style rule and every element that uses it).
3. Add a new section right after chapter 6's section: class "panel-extra", id "s6b", with an h2 "Chapter 6½" and one paragraph "Extra notes.".
4. The visit counter never goes up: `bumpVisits` stores the old number. Fix it so each call stores and returns one more than before.
