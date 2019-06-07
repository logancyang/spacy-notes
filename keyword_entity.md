
# How to Create Keyword List Entities in spaCy (v2.1)

Sometimes there is a need to create keyword entities from a list of known keywords, e.g. country names, species, brands, etc. The list can be large. This note shows how to create such entities in spaCy and make
it work with a trained NER model.

## Rule Based Matcher

`PhraseMatcher` is useful if you already have a large terminology list or gazetteer consisting of single or multi-token phrases that you want to find exact instances of in your data. As of spaCy v2.1.0, you can also match on the LOWER attribute for fast and case-insensitive matching.

`Matcher` is about individual tokens. For example, you can find a noun, followed by a verb with the lemma “love” or “like”, followed by an optional determiner and another token that’s at least ten characters long.

`PhraseMatcher` is what we need.

Say we have several brand names,

```
[u"Armani", u"Ralph Lauren", u"Monique Lhuillier", u"Norma Kamali"]
```

Assume we have some text messages in which we find these brand names. We apply the trained NER model on these messages to make predictions.
To make this case insensitive, use `attr="LOWER"`,

```
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher

nlp = English()
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(name) for name in [u"Armani", u"Ralph Lauren", u"Monique Lhuillier", u"Norma Kamali"]]
matcher.add("Brands", None, *patterns)

doc = nlp(u"armani and monique Lhuillier are both brands")
for match_id, start, end in matcher(doc):
    print("Matched based on lowercase token text:", doc[start:end])

# output:
# Matched based on lowercase token text: armani
# Matched based on lowercase token text: monique Lhuillier
```

It can even match number entities by shape, `attr="SHAPE"`, e.g. IP addresses.

## Combine with Model Prediction: Use Entity Ruler and Pattern File (v2.1)

`PhraseMatcher` doesn't address the need to combine rules with statistical models. The rules must have influence on the prediction process or they will have conflicts.

Citing the spaCy docs, "The entity ruler is designed to integrate with spaCy’s existing statistical models and enhance the named entity recognizer. **If it’s added before the `"ner"` component, the entity recognizer will respect the existing entity spans and adjust its predictions around it.** This can significantly improve accuracy in some cases. If it’s added after the `"ner"` component, the entity ruler will only add spans to the `doc.ents` if they don’t overlap with existing entities predicted by the model. To overwrite overlapping entities, you can set `overwrite_ents=True` on initialization."

```
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

# Before training
nlp = English()

"""This is the hard-coded ruler
ruler = EntityRuler(nlp)
patterns = [{"label": "ORG", "pattern": "Apple"},
            {"label": "GPE", "pattern": [{"lower": "san"}, {"lower": "francisco"}]}]
ruler.to_disk("./patterns.jsonl")
ruler.add_patterns(patterns)
"""

# Loading ruler from jsonl file
ruler = EntityRuler(nlp).from_disk("./patterns.jsonl")
nlp.add_pipe(ruler)

# Add NER training / transfer learning code here...

# At prediction time
doc = nlp(u"Apple is opening its first big office in San Francisco.")
print([(ent.text, ent.label_) for ent in doc.ents])
```

Question: Since the pattern file is a list of patterns, it must be slow to go through the list every time to check whether something is a brand. What's the solution?

## Case Study: Brand Entity

Brand is an example where the keyword list / pattern file can be really large. There are already many labeled brand entities in the training data so the model may or may not find correct brand entities at prediction time. In the case of an incorrect prediction, how do we leverage the rule-based method to correct it?

Note that we prefer adding the `EntityRuler` before the `"ner"` component to let the model respect the keyword list and adjust its predictions.

#### Case 1: Predicted entity is not in the keyword list and has no word overlap with any item in the list.

In this case, it is either a wrong prediction or a new brand entity correctly predicted but is not in the training data. These cases need to be logged and checked by a human. If confirmed it IS a correct new brand entity, it should be added to the brand keyword list.

#### Case 2: Predicted entity is not in the keyword list BUT has overlap with one or more items in the list.

If `EntityRuler` is used, the model prediction should be able to find the complete brand name in the text, so any such overlap should be the case where only part of the brand name is there in the text but no complete name from the brand list is present. This is sometimes OK, people don't necessarily call out the complete brand name but only refer to it with a short form. In other cases, this is a wrong prediction. Again, human check is preferred.

#### Case 3: Predicted entity is in the keyword list

This is the trivial case where the model is doing a perfect job.

(For more advanced usages involving dependency parsing, check [here](https://spacy.io/usage/rule-based-matching#models-rules-pos-dep) for examples. This is beyond the scope of this post about keyword list entities.)