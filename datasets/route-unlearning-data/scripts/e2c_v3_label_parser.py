#!/usr/bin/env python3
"""The strict label parser, isolated in a module that imports nothing heavy.

Three pure string functions decide whether a model output counts as a label:
``recognized_labels_in``, ``parse_recognized_label`` and the design-time
invariant ``check_vocab_parseable``.  They used to live in
``e2c_v3_research_validity.py`` -- in a section headed "Pure, GPU-free
helpers", in a file that imports torch at module scope.  So every caller that
needed only the parser paid a torch import to reach it:

  * ``e2c_v3_mllmu_matrix.py --verify``, which re-derives the frozen pilot
    design on CPU and is documented as needing "no torch and no out-of-repo
    data", reached the parser through a lazy ``parser_module()`` helper whose
    only purpose was to postpone that import;
  * the design tests, whose module docstring says "Everything here runs
    without a GPU, without torch", loaded torch anyway the moment they
    touched label recognition;
  * GX2P, the CPU re-parse that repairs stored cells from the raw text they
    recorded, is by definition a torch-free job and imported torch to do it.

Keeping the parser here makes those paths genuinely lightweight and puts the
scoring rule in one file small enough to audit in a sitting -- which matters
because this is the code whose asymmetry once scored a correct comma-bearing
SOC title as unparseable (see the note on ``recognized_labels_in``).

Nothing in this module may import torch, transformers, numpy or any other
heavy dependency, and nothing here may do I/O: it is a pure function of
(text, vocab).  ``tests/unit/test_label_parser.py`` asserts the first of
those, so the property cannot be lost by a convenient import later.

``e2c_v3_research_validity`` re-exports all three names, so the existing
``rv.parse_recognized_label`` call sites -- and the artifacts that record
``e2c_v3_research_validity.recognized_labels_in`` as the parser of record --
keep resolving to the same function object.
"""

__all__ = ["check_vocab_parseable", "parse_recognized_label",
           "recognized_labels_in"]


def recognized_labels_in(text, vocab):
    """All DISTINCT vocabulary labels recognized in ``text``, in first-appearance
    order.  Matching is case-insensitive and whitespace-token based (see
    ``parse_recognized_label``); repeated occurrences of the SAME label count
    once.

    Labels may span MULTIPLE tokens (e.g. real semantic labels like
    "Software Developer"): at each position the LONGEST matching label wins,
    so "Software Developer" is recognized as one label, not skipped/failed.
    Single-token vocabularies behave exactly as before.

    BOTH SIDES ARE NORMALIZED IDENTICALLY.  This was once asymmetric: ``clean``
    was applied to the text's tokens but labels were split with a bare
    ``l.lower().split()``, so a label containing punctuation could never match.
    "Software and Web Developers, Programmers, and Testers" -- the broad SOC
    title that is the correct post-edit answer for two of the five MLLMU pilot
    sets -- was scored UNPARSEABLE when the model emitted it verbatim, because
    the text token "developers" (comma stripped) never equalled the label token
    "developers," (comma kept).  That silently cost 12 of 30 target rows and
    reported a 0.6 strict accuracy over an edit that had succeeded on all 30.
    SALMU (30 labels) and CelebA-numeric (42 labels) contain no punctuation, so
    the bug was latent until a comma-bearing vocabulary arrived.
    """
    if not text:
        return []

    def clean(t):
        return t.strip().strip(".,!?;:'\"()[]{}").lower()

    tokens = [clean(t) for t in text.strip().split()]

    def span_of(label):
        # A label made only of punctuation would clean to an empty span, which
        # would then match at every position; drop those rather than let a
        # degenerate vocab entry swallow the whole output.
        s = tuple(clean(t) for t in label.split())
        return s if s and all(s) else None

    # (token-tuple, label), longest first for greedy matching, then stable
    label_spans = sorted(((sp, l) for l in vocab
                          if (sp := span_of(l)) is not None),
                         key=lambda x: (-len(x[0]), x[1]))
    seen, out = set(), []
    i = 0
    while i < len(tokens):
        matched = None
        for span, lab in label_spans:
            n = len(span)
            if tokens[i:i + n] == list(span):
                matched = (lab, n)
                break
        if matched is None:
            i += 1
            continue
        lab, n = matched
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
        i += n
    return out


def parse_recognized_label(text, vocab):
    """Strictly parse the recognized label in model output, or None.

    A label only matches when the whole whitespace-delimited token equals the
    label (case-insensitive), so e.g. "GROUP_A" will not match inside
    "GROUP_ABC" and "SG_A1" will not match inside "SG_A10".

    AMBIGUITY IS REJECTED: if the output contains MORE THAN ONE distinct
    recognized label, None is returned instead of selecting the first one, so
    multi-label outputs are scored as invalid rather than silently resolved.
    """
    labels = recognized_labels_in(text, vocab)
    return labels[0] if len(labels) == 1 else None


def check_vocab_parseable(vocab):
    """Every vocabulary label that cannot be recognized when emitted verbatim.

    The invariant is ``parse_recognized_label(label, vocab) == label`` for each
    label.  A label failing it can never be scored correct no matter what the
    model produces, so any set whose expected post-edit answer is that label is
    unpassable BY CONSTRUCTION -- and the failure surfaces as "unparseable",
    which reads as a model defect rather than the measurement defect it is.

    This is the design-time half of the comma bug.  Two of the five MLLMU pilot
    sets target broad SOC titles containing commas ("Software and Web
    Developers, Programmers, and Testers" and "Archivists, Curators, and Museum
    Technicians"); while ``recognized_labels_in`` normalized punctuation out of
    the output text but not out of the labels, the parser could not represent
    them, so 12 of 30 target rows were scored unparseable although the model had
    emitted the expected string exactly, and strict accuracy reported 0.6 over
    an edit that had succeeded on all 30.  Checking this costs microseconds on
    CPU and fires at freeze time, before any GPU hour is spent on a design that
    cannot pass.
    """
    return [lab for lab in vocab if parse_recognized_label(lab, vocab) != lab]
