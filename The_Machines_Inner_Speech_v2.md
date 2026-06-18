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
same dissociation observed in the lexical study between performed and
represented meaning, now appearing in motor control rather than ritual
register. That the pattern recurs across so distant a domain is the
strongest available indication that the distinction is structural rather
than an artifact of any single experiment 7.

#### The Benefits of the LLM as a Theoretical Instrument

Using the LLM as an instrument to "test" language theory offers unique
benefits for the field:

1.  **Validating the Watershed:** LLMs allow researchers to
    experimentally locate the "watershed" between internal
    representation and output 3. By using **steering vectors** to edit
    an NLA-recovered internal plan (e.g., changing a "rabbit" rhyme to a
    "mouse" rhyme), MI can prove that internal representations causally
    direct external output 2.

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
