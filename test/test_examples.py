"""Verification test suite ensuring all 3 paradigm example implementations work cleanly."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestExampleParadigms(unittest.TestCase):
    """Tests that the example code files for Scratch, RAG, and Adapter execute cleanly."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="aimlite_example_test_")
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_churn_scratch_paradigm(self):
        """Tests the Customer Churn scratch model pipeline."""
        from docs.examples.churn_scratch.data import TelecomChurnDataset
        from docs.examples.churn_scratch.model import ChurnClassifier
        from docs.examples.churn_scratch.trainer import ChurnTrainer
        from docs.examples.churn_scratch.evaluator import ChurnEvaluator
        from docs.examples.churn_scratch.inference import ChurnInference

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            csv_path = tmppath / "telecom_churn.csv"
            csv_path.write_text(
                "AccountWeeks,ContractRenewal,DataPlan,DataUsage,CustServCalls,DayMins,DayCalls,MonthlyCharge,OverageFee,RoamMins,Churn\n"
                "128,1,1,2.7,1,265.1,110,89.0,9.87,10.0,0\n"
                "107,1,0,0.0,1,161.6,123,37.0,9.78,13.7,1\n"
                "137,1,0,0.0,0,243.4,114,49.0,6.66,12.2,0\n"
                "84,1,0,0.0,2,299.4,71,62.0,5.45,6.6,1\n"
                "75,1,0,0.0,3,166.7,113,41.0,7.42,10.1,0\n"
                "118,0,0,0.0,4,223.4,98,57.0,8.7,6.3,1\n"
                "90,1,0,0.0,1,180.0,100,45.0,7.0,8.0,0\n"
                "60,0,1,1.5,5,210.0,90,70.0,12.0,9.0,1\n"
                "110,1,1,3.0,2,190.0,105,80.0,5.0,11.0,0\n"
                "95,0,0,0.0,3,140.0,85,35.0,8.0,14.0,1\n",
                encoding="utf-8",
            )

            ds = TelecomChurnDataset(source=csv_path)
            records = ds.load()
            self.assertEqual(len(records), 10)

            X, y = ds.to_arrays()
            self.assertEqual(len(X), 10)
            self.assertEqual(len(y), 10)

            model = ChurnClassifier()
            trainer = ChurnTrainer()
            evaluator = ChurnEvaluator()
            infer = ChurnInference()

            # Train
            res = trainer.fit(model, ds)
            self.assertEqual(res["status"], "completed")

            # Evaluate
            metrics = evaluator.evaluate(model, ds)
            self.assertIn("accuracy", metrics)

            # Inference
            pred = infer.run(model, records[0])
            self.assertEqual(pred["status"], "success")
            self.assertIn("churn_risk", pred)

    def test_knowledge_rag_paradigm(self):
        """Tests the Document & Knowledge RAG pipeline."""
        from docs.examples.knowledge_rag.data import KnowledgeDocsDataset
        from docs.examples.knowledge_rag.model import SupportDocRAG
        from docs.examples.knowledge_rag.trainer import IndexBuilderTrainer
        from docs.examples.knowledge_rag.inference import RAGInference

        ds = KnowledgeDocsDataset()
        docs = ds.load_documents()
        self.assertGreater(len(docs), 0)

        model = SupportDocRAG()
        trainer = IndexBuilderTrainer()
        infer = RAGInference()

        # Build index
        res = trainer.fit(model, ds)
        self.assertEqual(res["status"], "completed")

        # Inference query
        out = infer.run(model, {"query": "How does authentication work?"})
        self.assertEqual(out["status"], "success")
        self.assertIn("answer", out)
        self.assertIn("sources", out)

    def test_instruction_adapter_paradigm(self):
        """Tests the LoRA / PEFT instruction adapter pipeline."""
        from docs.examples.instruction_adapter.data import InstructionDataset
        from docs.examples.instruction_adapter.model import LoRAInstructionModel
        from docs.examples.instruction_adapter.trainer import AdapterInstructionTrainer
        from docs.examples.instruction_adapter.inference import AdapterInference

        ds = InstructionDataset()
        prompts = ds.format_prompts()
        self.assertGreater(len(prompts), 0)

        model = LoRAInstructionModel(lora_rank=8)
        trainer = AdapterInstructionTrainer()
        infer = AdapterInference()

        # Train adapter
        res = trainer.fit(model, ds)
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["paradigm"], "lora_adapter")

        # Inference
        out = infer.run(model, {"instruction": "Summarize this ticket."})
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["adapter_rank"], 8)
