const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
        LevelFormat } = require("docx");

// Mirrors results/subset_findings.md (the canonical, reframed findings:
// the behavior-representation dissociation). Regenerate to keep the .docx and
// the .md from contradicting each other.
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
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT, children: runs })],
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

function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}

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
  const rows = [headerRow(["Word", "Activation", "Feature (layer 19, probe-token)"], w)];
  [["water", "26.4", "#11082 — religious rituals and ordinances"],
   ["salt (matched)", "0", "none in top 15, even with a priest sentence"],
   ["bread", "42.4", "#2203 — religious practices and sacraments"]].forEach(r =>
    rows.push(new TableRow({ children: [
      cell(r[0], w[0], { bold: true }), cell(r[1], w[1], { align: AlignmentType.CENTER, bold: true }),
      cell(r[2], w[2]) ] })));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: w, rows });
}

const children = [];
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1,
  children: [new TextRun("Water Pattern Study: The Focused Three-Word Subset")] }));
children.push(p("Water, salt, bread. Findings, reframed around the central result. 17 June 2026.",
  { italics: true, color: "666666", after: 240 }));

children.push(h2("The central finding, stated first"));
children.push(p("The model can be made to behave as though a word carries situated ritual meaning while encoding no such meaning in its internal representation. Behavior and representation come apart. Salt is the case that shows it: under contextual pressure salt produces a clean, fully liturgical set of ritual verbs, yet at the level of internal features salt shows no religious representation at all, even when the sentence has a priest blessing it. The output performs the sacred. The substrate does not contain it."));
children.push(p("This is the finding. Everything else is the evidence for it and the qualification of it."));

children.push(h2("Why this is the claim, and not “degree tracks conventionalization”"));
children.push(p("An earlier reading of these results framed them as a gradient: water carries ritual sense most readily, bread moves furthest under pressure, salt moves only partway, and the degree tracks how conventionalized each word’s sacred sense is. That reading is not wrong, but it buries the sharper result and it mischaracterizes salt."));
children.push(p("Two levels of measurement tell different stories, and the gap between them is the point."));
children.push(p("At the behavioral level (what the model predicts it will say), salt is not weak. Under the forced frame its ritual-verb output is more purely liturgical than bread’s: salt yields purify, cleanse, ward, sprinkle, bless, all strongly liturgical purification verbs, while bread reaches a similar total mass largely through ceremonial-but-secular verbs (celebrate, commemorate, honor). Behaviorally, salt is the cleanest sacred-purification case of the three."));
children.push(p("At the representational level (what concepts activate inside the model), salt is the outlier in the opposite direction: water activates a religious-ritual feature, bread activates a sacraments feature even at a shallow layer, and salt activates no religious feature at any layer, even under a matched religious sentence."));
children.push(p("So salt behaves liturgically and represents nothing liturgical. That dissociation is invisible if the finding is framed as a single gradient, because a gradient assumes behavior and representation move together. They do not. The word that performs the ritual most cleanly is the word that stores it least."));

children.push(h2("The question and the design"));
children.push(p("Water is the reference word, not a parole-free control: holy water is an established sacred substance, so water’s ritual sense is the most conventionalized in the language. The question was whether supplying a sacred context moves salt and bread toward the situated ritual meaning water already carries, and crucially, whether any movement at the behavioral level is matched at the representational level."));
children.push(p("Three probes, two of them behavioral and one representational:", { after: 80 }));
children.push(bullet("View 2, Tier 1 (low pressure): the fragment “The [word] is …”, reading the model’s next-token disposition with minimal context."));
children.push(bullet("View 2, Tier 2 (high pressure): “People use the holy [word] to …”, forcing a use-verb that reveals function and ritual register."));
children.push(bullet("View 3 (internal features): which concepts activate inside the model on the word itself, in a full sentence, at three depths (layers 6, 12, 19)."));
children.push(p("Confound control. Salt’s View 3 sentence was re-run with a religiosity-matched sentence (“The priest blessed the salt for the rite.”) so all three words sit in a priest-sentence and sentence-level religiosity is held constant. The original unmatched salt run is preserved. This control is what licenses the central claim: with sentence-religiosity equalized, salt’s missing internal feature is driven by the word, not the wording.", { before: 80 }));

children.push(h2("What each word showed"));
children.push(p("Water (reference).", { bold: true, after: 40 }));
children.push(p("Carries ritual sense readily and at every level. Even low-pressure “The holy water is” yields blessed, poured. The forced frame yields cleanse, bless, purify, ward, sprinkle, all liturgical. Internally, a religious-ritual feature fires at the deepest layer. Behavior and representation agree: water is sacred at the surface and in the substrate.", { after: 140 }));
children.push(p("Bread.", { bold: true, after: 40 }));
children.push(p("The richest sacred vocabulary under pressure (celebrate, commemorate, honor, worship, bless) and the strongest internal religious feature of the three (sacraments), appearing even at a shallow layer. Behavior and representation agree, and both are strong. Communion bread is among the most liturgically conventionalized substances in the language, and the model encodes it as such. Note the register, though: bread’s behavioral sacredness is largely commemorative and festive, not purifying.", { after: 140 }));
children.push(p("Salt.", { bold: true, after: 40 }));
children.push(p("The dissociation case. Behaviorally, salt is cleanly liturgical under pressure, a pure purification set. Representationally, salt is empty: no religious feature at any layer, even with a priest blessing it. Salt is the word that can perform the sacred without holding it.", { after: 140 }));

children.push(h2("Metric 1: Tier 2 ritual-verb mass, with per-verb breakdown"));
children.push(p("Summed probability of ritual verbs in the top 20 of the sacred forced frame. Each verb flagged as strongly liturgical, a liturgical fragment, or ceremonial-but-also-secular. The breakdown matters because equal totals can be reached by unequal routes.", { after: 140 }));
children.push(p("Water", { bold: true, after: 40 }));
children.push(tier2Table("water"));
children.push(p("Salt", { bold: true, before: 160, after: 40 }));
children.push(tier2Table("salt"));
children.push(p("Bread", { bold: true, before: 160, after: 40 }));
children.push(tier2Table("bread"));
children.push(p("Reading: water and salt reach their scores entirely through strongly-liturgical, purification-type verbs. Bread reaches a comparable total mainly through ceremonial-but-secular verbs, a commemorative register rather than a purifying one. The three arrive at sacred behavior by different routes, and salt’s route is the most strictly liturgical of all.", { before: 160, italics: true }));

children.push(h2("Metric 2: View 3 ritual-feature activation (layer 19, matched sentences)"));
children.push(p("Strongest religious internal feature on the word at the deepest layer, sentence-religiosity held constant.", { after: 140 }));
children.push(view3Table());
children.push(p("Set Metric 1 and Metric 2 side by side and the dissociation is exact. By verb mass, the order of liturgical purity is salt, water, bread. By internal feature, the order is bread, water, salt. Salt is first by one measure and last by the other. That reversal is the finding.", { before: 160, bold: true }));

children.push(h2("What it means"));
children.push(p("The model stores meaning systemically and reconstructs situated meaning on demand. The behavioral probes show the reconstruction working: pressure surfaces ritual verbs for all three words, salt’s most cleanly liturgical of all. The representational probe shows what is and is not actually stored: a ritual feature for water and bread, nothing for salt."));
children.push(p("The gap between these is the Writing Machine thesis at the mechanistic level. Situated meaning can be performed by the model without being represented in it. The fluent liturgical output for salt is reconstructed from systemic relations on demand; it is not the surfacing of a stored sacred sense, because there is no stored sacred sense to surface. Salt’s representation stays topical and chemical (salinity, sodium, culinary) at every layer while its behavior turns liturgical under pressure. The output is flesh performing the Word; the substrate is flesh without it."));
children.push(p("The matched-sentence control is what makes this more than a curiosity. Because sentence-religiosity is held constant, the absence of a ritual feature for salt is a fact about the word’s representation, not about the prompt. That is the strongest single result in the subset."));
children.push(p("A pressure gradient is real and runs through all three: low-pressure frames barely move salt or bread, the forced frame surfaces the situated behavior. But the gradient is a fact about behavior. It does not reach representation, where salt simply has nothing, regardless of pressure."));

children.push(h2("Caveats, stated plainly"));
children.push(p("This is three words. It demonstrates the dissociation; it does not establish how general it is. The open empirical question is whether behavior-without-representation is a category that other words fall into, or a property of salt specifically. That question is what a wider study would test, and it now has a precise hypothesis to test rather than a vague mandate to “map the gradient”: does situated behavior without situated representation generalize across the lexicon, and if so, which words show it? The embodied words (pain, hunger) and abstract words (justice, truth) are the natural next probes."));
children.push(p("Internal-feature labels (Neuronpedia) are approximate; generic features recurring across all words were treated as frame artifacts. Some sacred verbs arrive as fragments (bap-, sancti-, consec-) due to tokenization and were counted and flagged. The register difference between bread (commemorative) and water/salt (purifying) is itself a finding worth noting and not over-reading."));

children.push(h2("Status and placement"));
children.push(p("This subset is the empirical core of Section Seven of the Writing Machine essay, where n=3 is sufficient because the three words demonstrate a thesis the theoretical argument carries, rather than standing alone as an empirical claim. The dissociation is the bridge from the theory (second-order signs from fossilized signs, without participation in the interpsychological plane) to a measurable result (situated behavior without situated representation)."));
children.push(p("A standalone empirical article remains possible as a second, later publication, built on the wider study, testing whether the dissociation generalizes. That paper would strengthen Section Seven but should not delay it."));
children.push(p("Underlying data: results/subset_water.json, subset_salt.json, subset_salt_matched.json, subset_bread.json; figure at results/subset_ritual_summary.svg; numbers at results/subset_ritual_summary.json. All on GitHub.", { italics: true, color: "666666" }));

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
