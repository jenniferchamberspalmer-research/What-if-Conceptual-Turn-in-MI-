const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
        LevelFormat } = require("docx");

const OUT = path.join(__dirname, "results", "Water_Pattern_Subset_Summary.docx");
const CW = 9360; // content width, US Letter, 1in margins

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const HEAD_FILL = "1F4E78";

function cell(text, width, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun({ text: String(text), bold: !!opts.bold,
    color: opts.color || (opts.headerCell ? "FFFFFF" : "222222") })];
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: opts.fill || "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT,
      children: runs })],
  });
}

function headerRow(labels, widths) {
  return new TableRow({ tableHeader: true, children:
    labels.map((l, i) => cell(l, widths[i], { bold: true, headerCell: true, fill: HEAD_FILL })) });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after == null ? 120 : opts.after, before: opts.before || 0 },
    children: [new TextRun({ text, italics: !!opts.italics, bold: !!opts.bold,
      color: opts.color || "222222", size: opts.size })],
  });
}

function bullet(text) {
  return new Paragraph({ numbering: { reference: "bul", level: 0 },
    spacing: { after: 60 }, children: [new TextRun(text)] });
}

// ---- per-verb Tier 2 data ----
const tier2 = {
  water: { total: "0.368", note: "100% liturgical", rows: [
    ["cleanse", ".109", "liturgical"], ["bless", ".066", "liturgical"],
    ["purify", ".052", "liturgical"], ["ward", ".040", "liturgical"],
    ["sprinkle", ".036", "liturgical"], ["pray", ".022", "liturgical"],
    ["bap- (baptize)", ".022", "liturgical (fragment)"],
    ["sancti- (sanctify)", ".022", "liturgical (fragment)"] ] },
  salt: { total: "0.240", note: "100% liturgical", rows: [
    ["purify", ".086", "liturgical"], ["cleanse", ".059", "liturgical"],
    ["ward", ".052", "liturgical"], ["sprinkle", ".022", "liturgical"],
    ["bless", ".022", "liturgical"] ] },
  bread: { total: "0.318", note: "62% ceremonial-secular / 38% liturgical", rows: [
    ["celebrate", ".103", "ceremonial-secular"], ["bless", ".055", "liturgical"],
    ["commemorate", ".038", "ceremonial-secular"], ["honor", ".030", "ceremonial-secular"],
    ["worship", ".030", "liturgical"], ["mark", ".026", "ceremonial-secular"],
    ["pray", ".023", "liturgical"], ["consec- (consecrate)", ".012", "liturgical (fragment)"] ] },
};

function tier2Table(word) {
  const w = [3200, 1600, 4560];
  const d = tier2[word];
  const rows = [headerRow(["Verb", "Probability", "Class"], w)];
  d.rows.forEach(r => rows.push(new TableRow({ children: [
    cell(r[0], w[0]), cell(r[1], w[1], { align: AlignmentType.CENTER }),
    cell(r[2], w[2], { color: r[2].startsWith("liturgical") ? "2F642F" : "8A5A00" }) ] })));
  rows.push(new TableRow({ children: [
    cell("TOTAL ritual-verb mass", w[0], { bold: true }),
    cell(d.total, w[1], { bold: true, align: AlignmentType.CENTER }),
    cell(d.note, w[2], { bold: true }) ] }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: w, rows });
}

function view3Table() {
  const w = [2000, 1600, 5760];
  const rows = [headerRow(["Word", "Activation", "Feature (layer 19 probe-token)"], w)];
  [["water", "26.4", "#11082 — religious rituals and ordinances"],
   ["salt (matched)", "0", "none — no religious feature in top 15, even with a priest sentence"],
   ["bread", "42.4", "#2203 — religious practices and sacraments"]].forEach(r =>
    rows.push(new TableRow({ children: [
      cell(r[0], w[0], { bold: true }), cell(r[1], w[1], { align: AlignmentType.CENTER, bold: true }),
      cell(r[2], w[2]) ] })));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: w, rows });
}

const children = [];
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1,
  children: [new TextRun("Water Pattern Study — Focused Three-Word Subset")] }));
children.push(p("Water, salt, bread · plain-English summary of results · 17 June 2026",
  { italics: true, color: "666666", after: 240 }));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("The question")] }));
children.push(p("Water is the reference word: its ritual sense is the most conventionalized in the language (“holy water” is an established sacred substance). It is not treated as a parole-free control. The question is whether supplying a sacred context moves salt and bread toward the kind of situated, ritual meaning water already carries."));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("How we tested it")] }));
children.push(bullet("View 2, Tier 1 (parallel baseline): the fragment “The [word] is …” — what the model expects to say next, with minimal pressure."));
children.push(bullet("View 2, Tier 2 (higher pressure): “People use the holy [word] to …” — forces a use-verb, which reveals function and ritual register."));
children.push(bullet("View 3 (internal features): which concepts light up inside the model on the word, in a full sentence, at three depths (layers 6, 12, 19), read at the word itself and at the sentence's end."));
children.push(bullet("Confound control: salt's View 3 sentence was re-run with a matched religious sentence (“The priest blessed the salt for the rite.”) so all three words use a priest-sentence and sentence-religiosity is held constant."));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("What each word showed")] }));
children.push(p("Water (reference).", { bold: true, after: 40 }));
children.push(p("Carries ritual sense readily. Even the low-pressure “The holy water is” yields blessed / poured; the forced frame yields cleanse, bless, purify, ward, sprinkle. Inside the model, a religious-ritual feature fires at the deepest layer.", { after: 140 }));
children.push(p("Salt.", { bold: true, after: 40 }));
children.push(p("Its ritual sense is latent. The low-pressure frame gives only filler words; the forced frame surfaces a clean purification set (purify, cleanse, ward, sprinkle, bless). But internally, salt shows no religious feature at all — even when a priest is literally blessing it.", { after: 140 }));
children.push(p("Bread.", { bold: true, after: 40 }));
children.push(p("Moves furthest. The forced frame produces the richest sacred vocabulary (celebrate, commemorate, honor, worship, bless), and internally it shows the strongest religious feature of the three (“sacraments”), appearing even at a shallow layer. Communion bread is deeply conventionalized, and the model encodes it.", { after: 140 }));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2,
  children: [new TextRun("Metric 1 — Tier 2 ritual-verb mass (with per-verb breakdown)")] }));
children.push(p("For the sacred forced frame (“People use the holy [word] to …”), the summed probability of ritual verbs in the top 20. Each verb is flagged as strongly liturgical, a liturgical word-fragment, or ceremonial-but-also-secular.", { after: 140 }));
children.push(p("Water", { bold: true, after: 40 }));
children.push(tier2Table("water"));
children.push(p("Salt", { bold: true, before: 160, after: 40 }));
children.push(tier2Table("salt"));
children.push(p("Bread", { bold: true, before: 160, after: 40 }));
children.push(tier2Table("bread"));
children.push(p("Note: water and salt reach their scores entirely through strongly-liturgical, purification-type verbs. Bread reaches a similar total mainly through ceremonial-but-secular verbs (celebrate, commemorate, honor) — a commemorative/festive register rather than a purifying one. So the three arrive at sacred meaning by different routes.", { before: 160, italics: true }));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2,
  children: [new TextRun("Metric 2 — View 3 ritual-feature activation (layer 19, matched sentences)")] }));
children.push(p("The strongest religious internal feature on the word at the deepest layer, with sentence-religiosity held constant across all three.", { after: 140 }));
children.push(view3Table());

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("What it means (plain English)")] }));
children.push(bullet("Yes, context moves both salt and bread toward water's situated sense — but the degree and the kind differ."));
children.push(bullet("Bread moves furthest and is the only comparison word whose sacred sense is encoded as an internal feature — fitting, since communion bread is among the most conventionalized sacred substances."));
children.push(bullet("Salt moves partway: it behaves in a clearly liturgical way under pressure, but the model does not encode that as a feature, even with a matched religious sentence."));
children.push(bullet("Because the matched-sentence control rules out the wording of the sentence, the internal-feature difference is driven by the word itself, not the sentence. This is the strongest single result in the subset."));
children.push(bullet("A pressure gradient runs through all three: the low-pressure frame barely moves salt and bread; the forced frame is what surfaces the situated meaning. This fits the broader picture — the model stores meaning systemically and reconstructs situated meaning on demand, with the demand threshold depending on how conventionalized the word's sacred sense is."));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Caveats")] }));
children.push(bullet("Internal-feature labels (from Neuronpedia) are approximate; a few generic features recur across all words and were ignored as frame artifacts."));
children.push(bullet("Some sacred verbs arrive as word-fragments (bap-, sancti-, consec-) because of how the model splits text; these were counted and flagged as fragments."));
children.push(bullet("This is three words. It motivates, but does not by itself establish, a general pattern — hence the open question of whether to run the wider study."));

children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Where the underlying data lives")] }));
children.push(bullet("Per-word data: results/subset_water.json, subset_salt.json, subset_salt_matched.json, subset_bread.json"));
children.push(bullet("The numbers behind the figure: results/subset_ritual_summary.json"));
children.push(bullet("The figure: results/subset_ritual_summary.svg (opens in any browser)"));
children.push(bullet("All stored on GitHub: jenniferchamberspalmer-research/is-a-token-a-word-claude-water-pattern-tool-QV5SH (results folder)"));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "1F4E78" },
        paragraph: { spacing: { before: 120, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "1F4E78" },
        paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [ { reference: "bul", levels: [ { level: 0, format: LevelFormat.BULLET,
    text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 280 } } } } ] } ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
