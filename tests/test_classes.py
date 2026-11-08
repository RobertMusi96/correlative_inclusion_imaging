import numpy as np
import pandas as pd

from cii.classes import assign_training_class, make_class_table, read_class_table, write_class_table


def particles():
    rows = [("N", "(Ti)N", "typified", "matched", True), ("N", "(Ti)N", "typified", "not_found", False),
            ("N", "(Nb,Ti)N", "typified", "matched", True), ("O", "(Al,Mg)O", "typified", "matched", True),
            ("not_typified", np.nan, "not_typified", "matched", True), ("rejected", np.nan, "rejected", "outside", False)]
    p = pd.DataFrame(rows, columns=["class_label", "Type", "label", "status", "training_ok"])
    p["ecd_um"] = 2.0
    p["Ti_wt"], p["N_wt"], p["Fe_wt"] = 40.0, 8.0, 50.0
    return p


def test_class_table_roundtrip_and_assignment(tmp_path):
    p = particles()
    t = make_class_table(p)
    assert len(t) == 5 and set(t.training_class) == {""}
    assert t.set_index("Type").loc["(Ti)N", "n_particles"] == 2
    t.loc[t.Type == "(Ti)N", "training_class"] = "TiN"
    t.loc[t.Type == "(Nb,Ti)N", "training_class"] = "TiN"
    t.loc[t.class_label == "rejected", "training_class"] = "artifact"
    write_class_table(t, tmp_path / "c.xlsx")
    back = read_class_table(tmp_path / "c.xlsx")
    tc = assign_training_class(p, back)
    assert list(tc) == ["TiN", "TiN", "TiN", "", "", "artifact"]
    # rebuilding with a new Type keeps the entered classes
    p2 = pd.concat([p, p.iloc[[3]].assign(Type="(Al,Ca)O")], ignore_index=True)
    t2 = make_class_table(p2, back)
    assert t2.set_index("Type").loc["(Ti)N", "training_class"] == "TiN"
    assert t2.set_index("Type").loc["(Al,Ca)O", "training_class"] == ""


def test_rules_and_min_ecd(tmp_path):
    from cii.classes import apply_classes, class_summary, read_rules, read_settings
    p = particles()
    p.loc[5, "status"] = "matched"; p.loc[5, "training_ok"] = True
    p["Ti_wt"] = [40, 40, 40, 0.2, 0.0, 3.0]
    p.loc[2, "ecd_um"] = 1.2
    t = make_class_table(p)
    t.loc[t.class_label == "N", "training_class"] = "TiN"
    t.loc[t.class_label == "rejected", "training_class"] = "Artifact"
    rules = pd.DataFrame([{"rule": "R1", "class_label": "rejected", "Type": "", "condition": "Ti_wt >= 1",
                           "training_class": "TiN", "comment": "Ti assigned to matrix"}])
    write_class_table(t, tmp_path / "c.xlsx", rules, {"min_ecd_um": 1.5})
    r, st = read_rules(tmp_path / "c.xlsx"), read_settings(tmp_path / "c.xlsx")
    assert st["min_ecd_um"] == 1.5 and r.loc[0, "condition"] == "Ti_wt >= 1"
    a = apply_classes(p, read_class_table(tmp_path / "c.xlsx"), r, st)
    assert list(a.training_class) == ["TiN", "TiN", "", "", "", "TiN"]
    assert a.class_source[5] == "rule R1" and a.class_source[2].startswith("below min_ecd")
    s = class_summary(p, a)
    assert s["overview"].loc["rejected", "TiN"] == 1
