# Water Pattern Study: The Focused Three-Word Subset

Water, salt, bread. Findings, reframed around the central result. 17 June 2026.

## The central finding, stated first

The model can be made to behave as though a word carries situated ritual meaning while encoding no such meaning in its internal representation. Behavior and representation come apart. Salt is the case that shows it: under contextual pressure salt produces a clean, fully liturgical set of ritual verbs, yet at the level of internal features salt shows no religious representation at all, even when the sentence has a priest blessing it. The output performs the sacred. The substrate does not contain it.

This is the finding. Everything else is the evidence for it and the qualification of it.

## Why this is the claim, and not "degree tracks conventionalization"

An earlier reading of these results framed them as a gradient: water carries ritual sense most readily, bread moves furthest under pressure, salt moves only partway, and the degree tracks how conventionalized each word's sacred sense is. That reading is not wrong, but it buries the sharper result and it mischaracterizes salt.

Two levels of measurement tell different stories, and the gap between them is the point.

At the behavioral level (what the model predicts it will say), salt is not weak. Under the forced frame its ritual-verb output is more purely liturgical than bread's: salt yields purify, cleanse, ward, sprinkle, bless, all strongly liturgical purification verbs, while bread reaches a similar total mass largely through ceremonial-but-secular verbs (celebrate, commemorate, honor). Behaviorally, salt is the cleanest sacred-purification case of the three.

At the representational level (what concepts activate inside the model), salt is the outlier in the opposite direction: water activates a religious-ritual feature, bread activates a sacraments feature even at a shallow layer, and salt activates no religious feature at any layer, even under a matched religious sentence.

So salt behaves liturgically and represents nothing liturgical. That dissociation is invisible if the finding is framed as a single gradient, because a gradient assumes behavior and representation move together. They do not. The word that performs the ritual most cleanly is the word that stores it least.

## The question and the design

Water is the reference word, not a parole-free control: holy water is an established sacred substance, so water's ritual sense is the most conventionalized in the language. The question was whether supplying a sacred context moves salt and bread toward the situated ritual meaning water already carries, and crucially, whether any movement at the behavioral level is matched at the representational level.

Three probes, two of them behavioral and one representational:

- **View 2, Tier 1 (low pressure):** the fragment "The [word] is …", reading the model's next-token disposition with minimal context.
- **View 2, Tier 2 (high pressure):** "People use the holy [word] to …", forcing a use-verb that reveals function and ritual register.
- **View 3 (internal features):** which concepts activate inside the model on the word itself, in a full sentence, at three depths (layers 6, 12, 19).

**Confound control.** Salt's View 3 sentence was re-run with a religiosity-matched sentence ("The priest blessed the salt for the rite.") so all three words sit in a priest-sentence and sentence-level religiosity is held constant. The original unmatched salt run is preserved. This control is what licenses the central claim: with sentence-religiosity equalized, salt's missing internal feature is driven by the word, not the wording.

## What each word showed

**Water (reference).** Carries ritual sense readily and at every level. Even low-pressure "The holy water is" yields blessed, poured. The forced frame yields cleanse, bless, purify, ward, sprinkle, all liturgical. Internally, a religious-ritual feature fires at the deepest layer. Behavior and representation agree: water is sacred at the surface and in the substrate.

**Bread.** The richest sacred vocabulary under pressure (celebrate, commemorate, honor, worship, bless) and the strongest internal religious feature of the three (sacraments), appearing even at a shallow layer. Behavior and representation agree, and both are strong. Communion bread is among the most liturgically conventionalized substances in the language, and the model encodes it as such. Note the register, though: bread's behavioral sacredness is largely commemorative and festive, not purifying.

**Salt.** The dissociation case. Behaviorally, salt is cleanly liturgical under pressure, a pure purification set. Representationally, salt is empty: no religious feature at any layer, even with a priest blessing it. Salt is the word that can perform the sacred without holding it.

## Metric 1: Tier 2 ritual-verb mass, with per-verb breakdown

Summed probability of ritual verbs in the top 20 of the sacred forced frame. Each verb flagged as strongly liturgical, a liturgical fragment, or ceremonial-but-also-secular. The breakdown matters because equal totals can be reached by unequal routes.

**Water — total 0.368, 100% liturgical**
cleanse .109, bless .066, purify .052, ward .040, sprinkle .036, pray .022, bap- (baptize) .022, sancti- (sanctify) .022

**Salt — total 0.240, 100% liturgical**
purify .086, cleanse .059, ward .052, sprinkle .022, bless .022

**Bread — total 0.318, 62% ceremonial-secular / 38% liturgical**
celebrate .103, bless .055, commemorate .038, honor .030, worship .030, mark .026, pray .023, consec- (consecrate) .012

Reading: water and salt reach their scores entirely through strongly-liturgical, purification-type verbs. Bread reaches a comparable total mainly through ceremonial-but-secular verbs, a commemorative register rather than a purifying one. The three arrive at sacred behavior by different routes, and salt's route is the most strictly liturgical of all.

## Metric 2: View 3 ritual-feature activation (layer 19, matched sentences)

Strongest religious internal feature on the word at the deepest layer, sentence-religiosity held constant.

| Word | Activation | Feature (layer 19, probe-token) |
|---|---|---|
| water | 26.4 | #11082 — religious rituals and ordinances |
| salt (matched) | 0 | none in top 15, even with a priest sentence |
| bread | 42.4 | #2203 — religious practices and sacraments |

Set Metric 1 and Metric 2 side by side and the dissociation is exact. By verb mass, the order of liturgical purity is salt, water, bread. By internal feature, the order is bread, water, salt. Salt is first by one measure and last by the other. That reversal is the finding.

## What it means

The model stores meaning systemically and reconstructs situated meaning on demand. The behavioral probes show the reconstruction working: pressure surfaces ritual verbs for all three words, salt's most cleanly liturgical of all. The representational probe shows what is and is not actually stored: a ritual feature for water and bread, nothing for salt.

The gap between these is the Writing Machine thesis at the mechanistic level. Situated meaning can be performed by the model without being represented in it. The fluent liturgical output for salt is reconstructed from systemic relations on demand; it is not the surfacing of a stored sacred sense, because there is no stored sacred sense to surface. Salt's representation stays topical and chemical (salinity, sodium, culinary) at every layer while its behavior turns liturgical under pressure. The output is flesh performing the Word; the substrate is flesh without it.

The matched-sentence control is what makes this more than a curiosity. Because sentence-religiosity is held constant, the absence of a ritual feature for salt is a fact about the word's representation, not about the prompt. That is the strongest single result in the subset.

A pressure gradient is real and runs through all three: low-pressure frames barely move salt or bread, the forced frame surfaces the situated behavior. But the gradient is a fact about behavior. It does not reach representation, where salt simply has nothing, regardless of pressure.

## Caveats, stated plainly

This is three words. It demonstrates the dissociation; it does not establish how general it is. The open empirical question is whether behavior-without-representation is a category that other words fall into, or a property of salt specifically. That question is what a wider study would test, and it now has a precise hypothesis to test rather than a vague mandate to "map the gradient": does situated behavior without situated representation generalize across the lexicon, and if so, which words show it? The embodied words (pain, hunger) and abstract words (justice, truth) are the natural next probes.

Internal-feature labels (Neuronpedia) are approximate; generic features recurring across all words were treated as frame artifacts. Some sacred verbs arrive as fragments (bap-, sancti-, consec-) due to tokenization and were counted and flagged. The register difference between bread (commemorative) and water/salt (purifying) is itself a finding worth noting and not over-reading.

## Status and placement

This subset is the empirical core of Section Seven of the Writing Machine essay, where n=3 is sufficient because the three words demonstrate a thesis the theoretical argument carries, rather than standing alone as an empirical claim. The dissociation is the bridge from the theory (second-order signs from fossilized signs, without participation in the interpsychological plane) to a measurable result (situated behavior without situated representation).

A standalone empirical article remains possible as a second, later publication, built on the wider study, testing whether the dissociation generalizes. That paper would strengthen Section Seven but should not delay it.

Underlying data: results/subset_water.json, subset_salt.json, subset_salt_matched.json, subset_bread.json; figure at results/subset_ritual_summary.svg; numbers at results/subset_ritual_summary.json. All on GitHub.
