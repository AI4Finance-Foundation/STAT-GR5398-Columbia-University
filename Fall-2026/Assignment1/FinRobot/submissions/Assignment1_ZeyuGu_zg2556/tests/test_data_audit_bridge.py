"""Isolated synthetic arithmetic fixtures, not experimental data."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from data_audit import ebitda_bridge,numeric


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.ref={"ticker":"AMD","sec_accession":"SYNTHETIC","period_end":"2000-12-31"}
        values={"revenue":100,"operating_income":20,"net_income":25,"tax":-2,"interest":1,"other_da":3,"acquisition_amortization":4}
        self.ingredients=[{"ticker":"AMD","accn":"SYNTHETIC","end":"2000-12-31","ingredient":k,"val":str(v),"json_pointer":"/synthetic/"+k} for k,v in values.items()]
        self.raw={"revenue":"100","operatingIncome":"20","netIncome":"25","incomeTaxExpense":"-2","interestExpense":"1","depreciationAndAmortization":"7","ebitda":"31"}
    def test_signed_tax_and_separate_amortization(self):
        result=ebitda_bridge(self.ref,self.raw,self.ingredients)
        self.assertEqual(result["status"],"RECONCILED_NET_INCOME_BASED_DEFINITION")
        self.assertEqual(result["sec_operating_ebitda"],27)
        self.assertEqual(result["sec_net_income_based_ebitda"],31)
        self.assertEqual(result["sec_nonoperating_excluding_interest"],4)
        self.assertEqual(result["unexplained_residual_against_sec_net_income_definition"],0)
    def test_component_discrepancy_not_excused_by_formula(self):
        self.raw.update(netIncome="26",ebitda="32")
        result=ebitda_bridge(self.ref,self.raw,self.ingredients)
        self.assertFalse(result["definition_reconciled"])
        self.assertEqual(result["status"],"FMP_NET_INCOME_FORMULA_MATCH_BUT_COMPONENT_DIFFERENCES")
    def test_missing_component_not_imputed(self):
        del self.raw["interestExpense"]
        result=ebitda_bridge(self.ref,self.raw,self.ingredients)
        self.assertFalse(result["definition_reconciled"])
        self.assertIsNone(result["fmp_net_income_recomputed"])
    def test_continuing_pre_equity_bridge_requires_all_extra_components(self):
        values={"continuing_income":23,"discontinued_income":2,"equity_income":1,"pretax_before_equity":20}
        self.ingredients += [{"ticker":"AMD","accn":"SYNTHETIC","end":"2000-12-31","ingredient":k,"val":str(v),"json_pointer":"/synthetic/"+k} for k,v in values.items()]
        self.raw.update(ebitda="28",netIncomeFromContinuingOperations="23",netIncomeFromDiscontinuedOperations="2",ebit="21")
        result=ebitda_bridge(self.ref,self.raw,self.ingredients)
        self.assertEqual(result["status"],"RECONCILED_CONTINUING_PRE_EQUITY_DEFINITION")
        self.assertEqual(result["residual_after_selected_definition"],0)
        self.raw["netIncomeFromDiscontinuedOperations"]="3"
        self.assertFalse(ebitda_bridge(self.ref,self.raw,self.ingredients)["definition_reconciled"])
    def test_zero_tax_preserved(self):
        self.raw.update(incomeTaxExpense="0",ebitda="33")
        result=ebitda_bridge(self.ref,self.raw,self.ingredients)
        self.assertEqual(result["fmp_net_income_recomputed"],33)
    def test_no_nonfinite_numbers(self):
        for value in ["nan","inf","-inf"]:
            with self.assertRaises(ValueError): numeric(value)


if __name__=="__main__":unittest.main()
