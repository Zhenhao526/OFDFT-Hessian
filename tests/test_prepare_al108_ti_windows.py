import tempfile
import unittest
from pathlib import Path

from scripts.prepare_al108_ti_windows import parse_components


class PrepareAl108TiWindowsTests(unittest.TestCase):
    def test_component_parser_accepts_optional_reference_virial(self):
        rows = [
            "MPN_TI_COMPONENTS step=1 lambda=0.5 U_MPN_eV=-10 "
            "U_REF_eV=-11 DELTA_U_eV=1 RMIN_A=2.3",
            "MPN_TI_COMPONENTS step=2 lambda=0.5 U_MPN_eV=-9 "
            "U_REF_eV=-10 W_REF_eV=-3.2 DELTA_U_eV=1 RMIN_A=2.2",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "running_md.log"
            path.write_text("\n".join(rows) + "\n")
            parsed = parse_components(path)
        self.assertEqual([row["step"] for row in parsed], [1, 2])
        self.assertEqual([row["U_target_eV"] for row in parsed], [-10.0, -9.0])
        self.assertEqual([row["delta_U_eV"] for row in parsed], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
