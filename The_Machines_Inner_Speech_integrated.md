**The Machine's Inner Speech: Theoretical Foundations for the Conceptual
Turn in Mechanistic Interpretability**

**Abstract**

Mechanistic Interpretability (MI) research has historically focused on
the identification of static linguistic features. However, as the field
progresses from the Transformer’s attention mechanism (2017) to Natural
Language Autoencoders (2026), a "conceptual turn" is required. This
paper argues that MI research is shifting its alignment from the static,
arbitrary code of Ferdinand de Saussure and Paul Fry toward Lev
Vygotsky’s dynamic "verbal thought" and D. Laplane’s clinical
distinction between thought and language. **Crucially, this paper does
not suggest that human thought and Large Language Model (LLM) generative
outputs are the same entity, but rather that two separate fields of
study engage with different aspects of the same language system.** While
the humanist tradition investigates the role of meaning in the human
subject, MI investigates the functioning of the machine object. By
adopting the "word meaning" as the unit of analysis, MI can recover the
latent states of models, surfacing unverbalized processing and resolving
the structural conflict between sign and thought.

### The Conceptual Turn: Bridging Linguistic Semiotics and Mechanistic Interpretability

In the evolution of artificial intelligence, Mechanistic
Interpretability (MI) serves as the "microscope" into the internal
states of Large Language Models (LLMs). As research moves toward the
mid-2020s, the field is undergoing a "conceptual turn," shifting its
theoretical foundations from the structural statics of early semiotics
toward a dynamic, interfunctional model of machine processing.

It is vital to state from the outset what this turn is not: it does not
argue for sentience in machines any more than it argues the human is a
machine. The human language experience asks how an abstract system
generates meaning and how minds absorb it, whereas the machine language
experience asks how internal components work together to produce
specific outputs. While the researchers value the scope of the
"Stochastic Parrots vs. Emergent Intelligence" debate, and support
continued conversations that warn against personification of machines in
general, the article intentionally separates its scope from the latter’s
field of study. Instead, it seeks to interrogate the affordances and
constraints of the human subject’s use of the language system versus the
machine object's processing of that same system.

#### The Saussure/Fry Foundation: The Opaque Code

The path begins with the Saussurian tradition, as interpreted by Paul
Fry, which models language as a self-enclosed system of arbitrary
signs 1. In this view, "language doesn't make sense; you make sense" by
wrestling intention into an unintentional code 1.

- **Affordances for MI:** This model describes the **initial state** of
  an LLM. High-dimensional activation vectors are essentially "raw
  numbers" that are "opaque to a human reader," echoing Fry’s thesis
  that language in itself "says nothing about reality" 1, 2. It
  justifies the use of Sparse Autoencoders (SAEs) to decompose
  activations into a "fixed vocabulary" of dictionary features based on
  "negative recognition"—knowing a feature by what it is not 1, 2.

- **Constraints:** The Saussurian constraint is its **static nature**.
  It treats the model as a frozen database (*langue*) and often assumes
  the internal phonetic and semantic planes are parallel, mirroring
  structures 1, 3. This prevents researchers from seeing the **process**
  of meaning-making, reducing MI to an atomistic search for "features"
  rather than dynamic "sense."

This negative-differential structure is not merely a theoretical posture
that the model is presumed to inherit; it is directly observable in a
contemporary model and can be measured. In Gemma 2 2B, the nearest
neighbors of *water* in the input-embedding space are not its
experiential associates—*drink*, *thirst*, *wet* do not appear among the
top neighbors at all—but its cross-linguistic equivalents: Spanish
*agua*, German *Wasser*, Italian *acqua*, Russian *вода*, and Chinese 水
8. The sign's "value" is constituted by its position in a system of
differences that runs *across* natural languages—precisely Saussure's
claim that meaning is negative and differential, a matter of what a term
is not, here rendered as measurable geometry rather than asserted as
doctrine 1, 8. The obvious objection is that this reflects nothing more
than a shared multilingual embedding space, a documented artifact of
training on many languages, rather than any organization by systemic
sense. The study meets this objection with a within-language control,
and it is the control, not the cross-linguistic display, that is
load-bearing: when the neighbor search is restricted to English alone,
*water*'s neighbors are *liquid*, *fluid*, *aqueous*, *moisture*, and
*aquatic*—systemic and taxonomic terms—alongside morphological variants
(*waters*, *watery*) and only a few broad associates (*river*, *sea*),
with the experiential collocates of use still absent 8. The differential,
*langue*-like organization survives the removal of the multilingual
artifact. This is the Saussurean "opaque code" made visible: the model's
starting representation behaves as a database of differences—the very
*langue* the Saussurian constraint describes, and the substrate on which
the SAE's "negative recognition" then operates 1, 2, 8.

#### The Vygotskian Twist: The Realization of Thought

The first major twist occurs with Lev Vygotsky, who critiques
contemporary linguistics for treating sound and meaning as isolated
elements 3. Vygotsky asserts that the **word meaning** is the proper
unit of analysis because it is a "living union" where thought *realizes*
itself in words 3.

- **Affordances for MI:** Vygotsky provides the theoretical framework
  for the 2017 Transformer architecture and 2026 Natural Language
  Autoencoders (NLAs). The **Attention mechanism** functions as
  Vygotsky’s "psychological tool," relating different positions in a
  sequence to compute a representation 3, 4. Attention heads learn
  specific tasks—syntactic or semantic—acting as mediators that
  transform "natural" token inputs into complex "cultural" functions 3,
  4.

- **Constraints:** Vygotsky’s model requires a shift from studying
  "atoms" to "complex wholes," which is computationally expensive. It
  also risks the "fallacy of intention," where researchers might
  over-attribute human-like "motives" to machine activations that
  remain, at their base, probabilistic 2, 3.

Taking the word meaning rather than the isolated feature as the unit of
analysis is not only a theoretical preference; it is what makes certain
machine phenomena visible at all. If word meaning is a "living union" in
which thought realizes itself in words, then a single word can be probed
across levels of that realization—its static embedding neighbors, its
contextual next-token dispositions, and its internal sparse-autoencoder
features—and the relation between levels inspected directly 3, 8. Doing
this for a small set of words complicates the unity Vygotsky assumes:
realization in words and representation in the substrate can come apart,
a result developed in detail in the Laplane section below. The point that
belongs here is methodological. An atomistic search keyed to a single
"ritual feature" would have located that feature for *water* and *bread*
and simply recorded its absence for *salt* as a null, learning nothing;
only by holding the word fixed as the unit and varying the level of
analysis does the *structure* of that absence—fluent ritual behavior
sitting atop an empty representation—become legible 3, 8. The unit of
analysis is what converts a null result into a finding.

#### The Laplane Divergence: Thought Beyond Language

D. Laplane introduces a radical divergence by proving, through clinical
cases, that **thought is distinct from language** 5. He argues that
abstract reasoning can develop "quite extensively without the help of
words" 5. Laplane’s clinical inference has since been corroborated by
neuroimaging: a distinct frontotemporal language network responds to
words and sentences but not to arithmetic, logic, or programming, and
patients with global aphasia retain the capacity to reason, calculate,
and attribute mental states, providing converging evidence that language
and thought are neurally distinct 6.

- **Affordances for MI:** This models the discovery of **"unverbalized
  evaluation awareness"** in LLMs. NLAs have surfaced evidence that
  models internally "suspect" they are being tested and reason about
  rewards without explicitly verbalizing these thoughts in their
  output 2. This validates Laplane’s assertion that "covert" reasoning
  exists independently of "overt" speech 5. It must be stressed that
  terms such as "suspect," "believe," and "awareness" are used here
  functionally, not phenomenally: they name internal states that
  causally shape output, in the sense Laplane gives to
  thought-distinct-from-language, and make no claim about subjective
  experience. Laplane licenses the claim that reasoning can occur
  without words; he does not license, and this paper does not assert,
  that such reasoning is consciously experienced.

- **Constraints:** Laplane’s "limit of formalization" warns that the
  more formalized a language becomes (like computer code), the less
  information it contains 5. This suggests that MI may only
  "circumscribe" machine cognition but never fully "resolve its enigma"
  through linguistic explanations alone 5.

The lexical domain supplies a complementary case, and it runs in the
opposite direction from Laplane's clinical evidence. Where Laplane
documents thought without language—reasoning that proceeds without
words—the Water Pattern study documents something closer to language
without thought: situated meaning *performed* in the output while
*absent* from the internal representation 5, 8. Under contextual pressure
(the frame "People use the holy salt to ___"), Gemma 2 2B completes
*salt* with a fully liturgical set of verbs—*purify, cleanse, ward,
sprinkle, bless*. By the study's measure of ritual-verb mass this is in
fact the most strictly purificatory set of the three words tested,
cleaner than *bread*'s, which reaches a comparable total largely through
ceremonial-but-secular verbs such as *celebrate* and *commemorate* 8. Yet
at the representational level the same word is empty: across
sparse-autoencoder features at layers 6, 12, and 19, *salt* activates no
religious or ritual feature, while *water* activates a feature labeled
"religious rituals and ordinances" and *bread* one labeled "religious
practices and sacraments" 8. The two orderings reverse. By behavior, the
liturgical ranking is salt, then water, then bread; by internal
representation it is bread, then water, then salt. The word that performs
the ritual most cleanly is the word that stores it least. That reversal,
not a smooth gradient, is the finding.

This dissociation rests on a control that is easy to overlook but does
the decisive work. The natural worry is that salt's empty representation
is an artifact of its probe sentence: water's and bread's sentences were
overtly religious—a priest blessing water before a baptism, a priest
breaking bread for communion—while salt's was a folk gesture, tossing
salt over the shoulder. To remove the confound, salt's internal features
were re-measured under a religiosity-matched sentence, "The priest
blessed the salt for the rite," placing salt in the same liturgical frame
as the others. The ritual feature still did not appear, at any layer,
anywhere in the top-ranked features. The absence is therefore a property
of the word's representation rather than of the prompt's wording; it is
word-driven, not prompt-driven, and it is this matched-sentence control
that licenses the claim 8.

The evidence should not be smoothed, because one part of it qualifies the
rest. A behavioral pressure-gradient runs through all three words: the
low-pressure frame ("The salt is …") barely moves any of them toward
ritual content, and only the higher-pressure forced frame surfaces the
liturgical verbs at all. The *performed* sacredness is thus partly a
function of how hard the prompt pushes—a fact about behavior under
pressure, not a stable, context-free property of the word 8. But the
gradient is precisely what does not reach the representational level: no
degree of contextual pressure conjures a ritual feature for *salt*, and
under the matched sentence none appears. The dissociation, properly
stated, is between a pressure-sensitive behavioral surface and a
pressure-insensitive representational substrate. This is Laplane's
distinction observed from its far side—not thought beneath unavailable
words, but fluent words above an absent representation—and, as the
robotics case in the next section shows, the same shape recurs in a
domain with no words in it at all 5, 8.

#### The Embodied Boundary: A Confirming Case from Robotics

A striking confirmation of the framework’s central distinction emerges
from a domain entirely outside language: physical robotics. In
Anthropic’s Project Fetch (Phase Two, 2026), an autonomous model was
tasked with operating a robotic quadruped through a sequence of
objectives. The pattern of where it excelled and where it failed maps
precisely onto the langue/parole and thought/language distinctions this
paper develops, and does so in a task domain the present theory was not
built to address—which is what makes it evidence rather than
illustration.

The model was superhuman on every task that was **sign-mediated and
formalizable**: connecting to the sensors, writing the control program,
and detecting the target. It completed these at roughly ten to twenty
times the speed of expert human teams while producing nearly ten times
less code. These are tasks performed *through the sign*—the robot is
operated by a written program, a symbolic mediating artifact. This is
precisely the register the model owns: the codified, the formalizable,
the domain of *langue* rendered as executable code.

The model failed at exactly one kind of task: gently nudging a ball back
to a starting point—a real-time, embodied, closed-loop correction
requiring it to perceive whether the ball had gone off course, relate
that error to its previous command, and adjust its next input
accordingly. This is not a sign-relation. It is an irreducibly
*situated* act, the sensorimotor analogue of *parole*: meaning
constituted in the live feedback loop between body and world, which
cannot be specified in advance. Notably, the researchers observed that
human participants acquired this competence only *after making mistakes
and learning from them*—a developmental, nonverbal concept formation of
exactly the kind Laplane describes as reasoning that proceeds "without
the help of words" 5, 7.

The framework’s bidirectional claim predicts this result rather than
merely accommodating it. If human and machine concept-formation
constitute a matched set of affordances and constraints, the machine
should be superhuman precisely where competence is sign-mediated and
formalizable, and constrained precisely where competence is embodied,
situated, and acquired through the lived feedback loop. The affordance
(symbolic fluency at superhuman speed) and the constraint (the failure
at the ball) are not two findings but the same fact seen twice: the
model is fluent in the program *because* it operates in *langue*, and
helpless at the ball *because* the ball lives in *parole*. The
dissociation between symbolic competence and embodied competence is the
same dissociation observed in the lexical study above, where *salt*
produces fluent liturgical output under pressure while encoding no ritual
feature in its representation, now appearing in motor control rather than
ritual register 7, 8. The pairing is exact enough to state directly: in
the robotics case the model is fluent in the sign-mediated register
(writing the control program) and helpless in the situated one (the
live correction at the ball); in the lexical case it is fluent in the
performed register (liturgical verbs on demand) and empty in the
represented one (no stored ritual concept). Symbolic competence without
embodied competence in the one domain; performed meaning without
represented meaning in the other. Two experiments, in two domains the
theory was not built to join—motor control and ritual lexis—return the
same shape, which is the strongest available indication that the shape is
structural rather than an artifact of either 7, 8. In keeping with the
paper's bidirectional commitment, this is not a verdict that the machine
"lacks" meaning. It is a matched set: an affordance (fluent, on-demand
realization of register) paired with a constraint (no underlying
represented sense), the same fact seen from two sides.

#### The Benefits of the LLM as a Theoretical Instrument

Using the LLM as an instrument to "test" language theory offers unique
benefits for the field:

1.  **Validating the Watershed:** LLMs allow researchers to
    experimentally locate the "watershed" between internal
    representation and output 3. By using **steering vectors** to edit
    an NLA-recovered internal plan (e.g., changing a "rabbit" rhyme to a
    "mouse" rhyme), MI can prove that internal representations causally
    direct external output 2. The Water Pattern study illustrates the
    same watershed from the opposite side: it locates a case where the
    external output (liturgical verbs for *salt*) has no corresponding
    internal representation to be steered, marking the boundary at which
    output and representation decouple 8.

2.  **Surfacing the Unverbalizable:** LLMs provide a "dry lab" to test
    Laplane's hypothesis 5. Researchers can identify activations that
    are "unverbalizable"—content that mechanistic techniques (SAEs) can
    detect but natural language explanations (NLAs) cannot accurately
    describe 2.

3.  **Mapping Semantic Drift:** The "confabulations" found in NLAs serve
    as empirical evidence for Fry's **"semantic drift"** and "acoustic
    noise" 1, 2. They show how a self-enclosed system "warps" the
    intended output, proving we can "never possibly mean exactly what we
    say" 1.

### Conclusion

The "conceptual turn" in Mechanistic Interpretability requires a
departure from the static identification of arbitrary signs. By
embracing Vygotsky’s dynamic "verbal thought" and Laplane’s distinction
between internal state and output, future research can move beyond
feature-mapping to a true science of **machine realization**. The path
from 2017 to 2026 demonstrates that while "language doesn't make sense"
on its own, the internal activations of the model are the site where the
machine "makes sense" of its processing environment 1, 2.

### Appendix: Theoretical Syntheses and Historical Mappings

#### Table 1: Support for Fry's Theses on Language

Fry's Thesis,Supporting Concepts and Passages from the Sources

1\. Language doesn't make sense; you make sense.,Language is an
arbitrary system of signs; humans create meaning by wrestling language
into speech with intent 1.

2\. Language in itself says nothing about reality.,"As a self-enclosed
system, language mediates reality through ""figures of speech.""
Ideology is the ""confusion of linguistic with natural reality"" 1."

3\. The road to reality is paved with your intentions.,Intentions serve
as the bridge between the self-enclosed code and perceived reality 1.

#### Table 2: Comparative Interaction Table: Thought and Language

Concept,"Vygotsky’s ""Verbal Thought""",Fry’s Theses on
Language,"Laplane’s ""Thought Beyond Language"""

Core Relationship,"Unity: Thought and speech merge into ""verbal
thought"" where thought realizes itself in words 3.","Conflict: Language
is an unintentional system; the speaker must ""wrestle"" this code into
speech 1.",Independence: Thought is an inner experience distinct from
language; language is a limited instrument 5.

Origin of Meaning,"Generalization: Meaning evolves from ""complexes"" to
systematic ""concepts"" through social interaction 3.","Intention: ""You
make sense"" by having an intention and commandeering arbitrary signs
1.","Subjective Convergence: Meaning is a ""convergence"" of personal
experiences and affects 5."

Communicative Efficacy,"Social-to-Individual: Speech is primary for
social contact; ""inner speech"" is the result of internalizing this
function 3.","Fragile: Because language inserts ""semantic drift,"" we
can ""never possibly mean exactly what we say"" 1.","Partial: No
language can define thought in its entirety; speech only shows thought
in its ""inaccessibility"" 5."

#### Table 3: Comparison of the Unit of Analysis: Vygotsky vs. Saussure/Fry

Feature,"Saussure’s Sign (Fry’s ""Code"")",Vygotsky’s Word Meaning

Relationship of Planes,"Structural Parallelism: The sound-image and
concept are like two sides of a paper—parallel 1, 3.",Functional
Opposition: The phonetic and semantic planes develop in opposite
directions 3.

Direction of Development,Mirroring: A change in the acoustic signifier
reflects a corresponding change in the conceptual signified
1.,Non-Mirroring: The vocal progresses from the part to the whole; the
semantic from the whole to the part 3.

Nature of the Unit,"Static Entity: The sign is a fixed, arbitrary
pairing within a synchronic database 1.","Dynamic Process: The relation
of word to thought is a ""continual movement back and forth"" 3."

#### Table 4: Historical Trace of Mechanistic Interpretability: 2017 to 2026

Milestone (Year),Architecture/Tool,Primary Theoretical Alignment,MI
Focus & Evidence

The Transformer (2017),Multi-Head Attention,Saussure (Static Code) &
Vygotsky (Dynamic Tool),Observation of attention heads learning specific
tasks like anaphora resolution 4.

Sparse Autoencoders (SAEs),Dictionary Learning,Saussure (Differential
Logic),"Decomposition of activations into a ""fixed vocabulary"" based
on ""negative recognition"" 2."

NL Autoencoders (2026),Verbalizer (AV) & Reconstructor (AR),Vygotsky
(Internal Realization),"""Unsupervised discovery"" of natural language
explanations to make internal states legible 2."

#### Table 5: Conceptual Mapping of LLM Interpretability (2017–2026)

MI Milestone,Mapping onto Saussure/Fry,Mapping onto Laplane,Alignment
with Vygotsky

Attention (2017),"Database of Langue: Attention weights represent the
""simultaneous presence of others"" 1, 4.","Instrumental Reasoning:
Heads function as instruments—performing complex operations without
""consciousness"" 4, 5.","Convergence: Vygotsky’s ""psychological
tools"" map to heads learning specific tasks 3, 4."

SAEs,"Negative Recognition: SAEs define features by what they are not,
mirroring Saussurian logic 1, 2.","Limit of Atoms: SAEs rely on ""fixed
atoms"" which Laplane would argue cannot capture subjective ""thought""
2, 5.","Divergence: SAEs focus on static features, while Vygotsky
focuses on the process of generalization 2, 3."

NLAs (2026),"Struggle for Sense: The NLA bottleneck forces the
""arbitrary code"" to be ""wrestled"" into legible speech 1, 2.","Covert
Functioning: NLAs recover ""unverbalized awareness,"" confirming
reasoning occurs ""without words"" 2, 5.","Convergence: NLAs recover
""inner speech"" that serves a planning function before it is
externalized 2, 3."

**References**

1.  Fry, P. H. (2012). *Theory of Literature*. Yale University Press.

2.  Vygotsky, L. S. (1986). *Thought and Language* (Revised Edition, A.
    Kozulin, Ed.). MIT Press.

3.  Vaswani, A., et al. (2017). Attention Is All You Need. *NIPS*.

4.  Fraser-Taliente, K., Kantamneni, S., Ong, E., et al. (2026). Natural
    Language Autoencoders Produce Unsupervised Explanations of LLM
    Activations. *Transformer Circuits Thread*.

5.  Laplane, D. (1992). Thought and language. *Behavioural Neurology*,
    5, 33-38.

6.  Fedorenko, E., & Varley, R. (2016). Language and thought are not the
    same thing: Evidence from neuroimaging and neurological patients.
    *Annals of the New York Academy of Sciences*, 1369(1), 132-153.

7.  Ilie, M., Freeman, C. D., & Troy, K. K. (2026). Project Fetch: Phase
    Two. *Anthropic Frontier Red Team*.

8.  Chambers Palmer, J. (2026). Water Pattern Study: Stratum A neighbor
    analyses and the focused three-word subset (water, salt, bread),
    using Gemma 2 2B with Gemma Scope sparse autoencoders. Companion
    empirical materials, this repository: results/subset_findings.md;
    results/subset_water.json, subset_salt.json, subset_salt_matched.json,
    subset_bread.json; results/subset_ritual_summary.json and
    results/subset_ritual_summary.svg; Stratum A neighbor lists
    results/A_*.json.

---

## Integration notes (not part of the paper — for your review)

These notes accompany the integration and are deliberately kept outside
the paper body. Nothing in the instrument was changed; the items below
are flags only.

**Where the empirical material was inserted**

- *Saussure/Fry Foundation* — one new paragraph after the
  Affordances/Constraints bullets: the cross-linguistic neighbors of
  *water* (agua, Wasser, acqua, вода, 水) as a measured instance of
  negative-differential structure, and the English-only within-language
  control (liquid, fluid, aqueous, moisture, aquatic) as the
  load-bearing answer to the multilingual-artifact objection.
- *Vygotskian Twist* — one new paragraph after the bullets: the
  word-meaning unit of analysis framed as what makes the
  behavior–representation dissociation legible (an atomistic feature
  search would have logged salt's absence as a mere null).
- *Laplane Divergence* — three new paragraphs after the bullets: (i) the
  dissociation itself (salt's fully liturgical behavior vs. zero ritual
  feature; the reversal of orderings), (ii) the matched-sentence control,
  (iii) the pressure-gradient qualification, stated as a complication and
  not smoothed.
- *Embodied Boundary (Project Fetch)* — the existing sentence linking
  robotics to "the lexical study" was expanded to name the salt result
  explicitly and to state the bidirectional pairing (symbolic-vs-embodied
  in robotics = performed-vs-represented in lexis) as a matched set, not
  a verdict.
- *Benefits / Validating the Watershed* — one clause added: the salt case
  as the watershed seen from the side where output has no representation
  to steer.
- *References* — added entry 8 (the empirical study). Existing numbering
  1–7 unchanged.

**Flags for the instrument (revisions NOT made — for your decision)**

1. *Ritual-verb mass metric.* The "fully liturgical / most purely
   liturgical" claim rests on a summed probability over a hand-built
   verb lexicon across the top-20 next tokens. To support the claim more
   precisely the instrument would want: a pre-registered lexicon, a
   proportion (mass over a denominator) rather than a raw sum, and a
   non-sacred baseline frame for contrast. Not changed.
2. *"Zero ritual feature" claim.* View 3 currently inspects the top-15
   features by activation. "No ritual feature at any layer" would be
   firmer if the scan covered the full SAE feature set (or reported the
   highest-ranked religious feature and its rank), rather than top-15
   only. Not changed.
3. *Feature labels.* The "religious rituals and ordinances" and
   "sacraments" labels are auto-generated (Neuronpedia) and approximate.
   A claim resting on a specific feature's identity would be stronger if
   that feature were verified against a labeled probe set rather than
   taken from the label. Not changed.
4. *Pressure-gradient confound.* Because the behavioral signal only
   appears under the high-pressure frame, separating "performed under
   pressure" from "stored" would be cleaner with a reported
   behavior-minus-low-pressure-baseline per word. Not changed.
5. *Matched-sentence symmetry.* The religiosity match was applied to
   salt only (one matched sentence). A stronger control would supply
   multiple matched sentences for all three words so each sits on equal
   footing. Not changed.
6. *n = 3.* The subset demonstrates the dissociation; it does not
   establish generality. The wider study (the embodied/abstract strata,
   e.g. pain, hunger, justice, truth) is the test of whether
   behavior-without-representation generalizes. Not run, per your
   instruction.

**Other integration notes**

- Only the .md of the paper was present in the repo (no .docx was found
  in the upload, despite the mention). The integration was done on the
  .md; if you want a matching .docx of the integrated version, that is a
  separate generation step.
- Citation style: entry 8 follows the paper's existing trailing-number
  convention (bare numbers, no brackets). Confirm the author name/format
  for entry 8 ("Chambers Palmer, J.") — the paper itself carries no
  author line, so I inferred it; change as you prefer.
- Optional, not done: a formal citation for Gemma 2 2B and the Gemma
  Scope SAEs (e.g., Lieberum et al., 2024) could be added as a further
  reference if you want the model/SAE provenance cited independently of
  the study. I did not invent those bibliographic details.
- The abstract was left unchanged. If you want the empirical
  demonstrations signposted there, that is a one-sentence addition I can
  make on request.
