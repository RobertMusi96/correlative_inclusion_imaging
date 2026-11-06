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
