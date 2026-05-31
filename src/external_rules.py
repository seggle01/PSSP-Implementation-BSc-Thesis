import re

######### Salamov-Solovyev Q3 post-processing rules #########

# Rule 1: remove implausibly short helices
def rule_1_Q3(predictions: str) -> str:
    # EHE → EEE (single H between strands becomes strand)
    result = re.sub(r"(?<=E)H(?=E)", "E", predictions)
    # Otherwise convert all 1-2 residue helices to coil
    result = re.sub(r"(?<!H)H{1,2}(?!H)", lambda m: "C" * len(m.group()), result)
    return result

# Rule 2: remove isolated single-residue strands
def rule_2_Q3(predictions: str) -> str:
    return re.sub(r"(?<!E)E(?!E)", "C", predictions)

# Rule 3: convert HEEH (strand-of-2 flanked by helices) to HHHH
def rule_3_Q3(predictions: str) -> str:
    return re.sub(r"HEEH", "HHHH", predictions)

######### Combined application #########
def apply_all_q3_rules(s: str) -> str:
    """Apply all three Q3 rules sequentially: rule1 → rule2 → rule3."""
    return rule_3_Q3(rule_2_Q3(rule_1_Q3(s)))
