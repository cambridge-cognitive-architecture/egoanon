"""The screen a line of model text must pass before it is kept: an allow-list.

A line is kept only if every word names an everyday object on this list. That bounds
what can be said. It does not make it true: the model can name objects that are not there.
"""
from __future__ import annotations

import re

MAX_WORDS, MIN_WORDS = 6, 2

# No people, bodies, clothing, conditions, quantities, colours, brands or places.
# Keep it that way when adding a word, and avoid words that are also common names.
OBJECT_WORDS = frozenset("""
    door doors doorway window windows wall walls floor ceiling stairs step steps railing
    pillar column
    desk desks table tables counter counters worktop bench benches chair chairs stool
    stools sofa couch
    cabinet cabinets cupboard cupboards drawer drawers shelf shelves shelving rack racks
    locker lockers bookcase furniture
    bed beds trolley trolleys cart carts mattress pillow pillows
    blanket blankets sheet sheets curtain curtains blind blinds partition
    computer computers monitor monitors screen screens laptop laptops keyboard
    keyboards mouse printer printers scanner phone phones telephone television projector
    camera speaker
    cable cables wire wires socket sockets plug switch switches panel panels device
    devices machine machines equipment supplies tools
    sink tap bin bins bucket mop dispenser dispensers towel towels tissue tissues
    soap bottle bottles cup cups mug mugs tray trays box boxes container containers
    basket baskets bag bags
    paper papers folder folders binder binders clipboard clipboards book books pen pens
    sign signs poster posters noticeboard whiteboard clock lamp lamps light lights
    lighting fan radiator vent pipe pipes pole poles stand stands wheels
    plant plants carpet rug tiles
    office filing storage water waste
""".split())

# Rejected by the list anyway. Named separately so that the reason is reported.
NUMBER_WORDS = frozenset("""
    zero nought nil one two three four five six seven eight nine ten eleven twelve
    thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty
    fifty sixty seventy eighty ninety hundred thousand million billion dozen half quarter
    single double triple first second third fourth fifth sixth seventh eighth ninth tenth
""".split())

# ASCII words separated by one space, or by one comma and one space. Nothing else.
LINE = re.compile(r"[A-Za-z]+(?:,? [A-Za-z]+)*")


def screen(line: str) -> list[str]:
    """Reasons the line must not be kept; empty means it passed.

    The reasons never quote the line.
    """
    reasons = []
    if re.search(r"\d", line):
        reasons.append("contains a digit")
    if not line.isascii():
        reasons.append("non-Latin or non-ASCII text")
    if not LINE.fullmatch(line):
        reasons.append("joined tokens or other characters")
    words = re.findall(r"[^\W\d_]+", line)  # letters of any script
    if any(w.lower() in NUMBER_WORDS for w in words):
        reasons.append("spelled-out number")
    if any(w.lower() not in OBJECT_WORDS for w in words):
        reasons.append("word not on the object list")
    # Only the first word may have a capital, and only as its first letter.
    if any(w != w.lower() and (i > 0 or w != w.capitalize()) for i, w in enumerate(words)):
        reasons.append("unexpected capital")
    if len(words) > MAX_WORDS:
        reasons.append("too long")
    if len(words) < MIN_WORDS:
        reasons.append("too short")
    return reasons
