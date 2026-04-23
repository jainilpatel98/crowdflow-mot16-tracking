import math

from Metrics import Metrics


class MOTMetrics(Metrics):
    """Wrapper around the MOTChallenge MATLAB evaluator.

    Use this after you have:
    1. A tracker prediction file in MOTChallenge text format.
    2. A matching GT file.
    3. The MATLAB MOT devkit available locally.

    This class does not run tracking itself. It only computes metrics from
    already-generated result files.
    """

    def __init__(self, seqName=None):
        super().__init__()
        self.seqName = seqName if seqName else 0

        # Standard CLEAR MOT metrics.
        self.register(name="MOTA", formatter="{:.2f}".format)
        self.register(name="MOTP", formatter="{:.2f}".format)
        self.register(name="MOTAL", formatter="{:.2f}".format, write_mail=False)

        # Identity-based metrics.
        self.register(name="IDF1", formatter="{:.2f}".format)
        self.register(name="IDP", formatter="{:.2f}".format)
        self.register(name="IDR", formatter="{:.2f}".format)
        self.register(name="IDTP", formatter="{:.0f}".format, write_mail=False)
        self.register(name="IDFP", formatter="{:.0f}".format, write_mail=False)
        self.register(name="IDFN", formatter="{:.0f}".format, write_mail=False)

        # Detection quality.
        self.register(name="recall", display_name="Rcll", formatter="{:.2f}".format)
        self.register(name="precision", display_name="Prcn", formatter="{:.2f}".format)
        self.register(name="tp", display_name="TP", formatter="{:.0f}".format)
        self.register(name="fp", display_name="FP", formatter="{:.0f}".format)
        self.register(name="fn", display_name="FN", formatter="{:.0f}".format)

        # Trajectory quality.
        self.register(name="MTR", formatter="{:.2f}".format)
        self.register(name="PTR", formatter="{:.2f}".format)
        self.register(name="MLR", formatter="{:.2f}".format)
        self.register(name="MT", formatter="{:.0f}".format)
        self.register(name="PT", formatter="{:.0f}".format)
        self.register(name="ML", formatter="{:.0f}".format)

        # Other summary values.
        self.register(name="F1", display_name="F1", formatter="{:.2f}".format, write_mail=False)
        self.register(name="FAR", formatter="{:.2f}".format)
        self.register(name="total_cost", display_name="COST", formatter="{:.0f}".format, write_mail=False)
        self.register(name="FM", formatter="{:.0f}".format)
        self.register(name="fragments_rel", display_name="FMR", formatter="{:.2f}".format)
        self.register(name="id_switches", display_name="IDSW", formatter="{:.0f}".format)
        self.register(name="id_switches_rel", display_name="IDSWR", formatter="{:.1f}".format)
        self.register(name="n_gt_trajectories", display_name="GT", formatter="{:.0f}".format, write_mail=False)
        self.register(name="n_tr_trajectories", display_name="TR", formatter="{:.0f}".format, write_db=False, write_mail=False)
        self.register(name="total_num_frames", display_name="TOTAL_NUM", formatter="{:.0f}".format, write_mail=False, write_db=False)
        self.register(name="n_gt", display_name="GT_OBJ", formatter="{:.0f}".format, write_mail=False, write_db=False)
        self.register(name="n_tr", display_name="TR_OBJ", formatter="{:.0f}".format, write_mail=False, write_db=False)

    def compute_clearmot(self):
        """Compute secondary summary metrics from raw counts."""
        if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
            self.recall = 0.0
            self.precision = 0.0
        else:
            self.recall = (self.tp / float(self.tp + self.fn)) * 100.0
            self.precision = (self.tp / float(self.fp + self.tp)) * 100.0

        if (self.recall + self.precision) == 0:
            self.F1 = 0.0
        else:
            self.F1 = 2.0 * (self.precision * self.recall) / (self.precision + self.recall)

        self.FAR = "n/a" if self.total_num_frames == 0 else (self.fp / float(self.total_num_frames))
        self.MOTA = -float("inf") if self.n_gt == 0 else (1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)) * 100.0
        self.MOTP = 0 if self.tp == 0 else (1 - self.total_cost / float(self.tp)) * 100.0

        if self.n_gt != 0:
            if self.id_switches == 0:
                self.MOTAL = (1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)) * 100.0
            else:
                self.MOTAL = (1 - (self.fn + self.fp + math.log10(self.id_switches)) / float(self.n_gt)) * 100.0

        if self.recall != 0:
            self.id_switches_rel = self.id_switches / self.recall
            self.fragments_rel = self.FM / self.recall
        else:
            self.id_switches_rel = 0
            self.fragments_rel = 0

        id_precision = self.IDTP / (self.IDTP + self.IDFP) if (self.IDTP + self.IDFP) else 0
        id_recall = self.IDTP / (self.IDTP + self.IDFN) if (self.IDTP + self.IDFN) else 0
        self.IDF1 = 0 if (self.n_gt + self.n_tr) == 0 else 2 * self.IDTP / (self.n_gt + self.n_tr)
        self.IDP = id_precision * 100
        self.IDR = id_recall * 100
        self.IDF1 = self.IDF1 * 100

        if self.n_gt_trajectories == 0:
            self.MTR = 0.0
            self.PTR = 0.0
            self.MLR = 0.0
        else:
            self.MTR = self.MT * 100.0 / float(self.n_gt_trajectories)
            self.PTR = self.PT * 100.0 / float(self.n_gt_trajectories)
            self.MLR = self.ML * 100.0 / float(self.n_gt_trajectories)

    def compute_metrics_per_sequence(self, sequence, pred_file, gt_file, gtDataDir, benchmark_name):
        """Call the MATLAB MOT evaluator for one sequence.

        Parameters
        ----------
        sequence:
            Sequence name such as ``MOT16-02``.
        pred_file:
            Path to the tracker output text file in MOTChallenge format.
        gt_file:
            Path to the sequence GT text file.
        gtDataDir:
            Root directory expected by the MATLAB devkit.
        benchmark_name:
            Usually ``MOT16``.
        """
        import matlab.engine

        try:
            eng = matlab.engine.start_matlab()
            print("MATLAB successfully connected")
        except Exception as exc:
            raise Exception("MATLAB could not connect. Install MATLAB Engine for Python and confirm MATLAB is available.") from exc

        eng.addpath("matlab_devkit/", nargout=0)
        results = eng.evaluateTracking(sequence, pred_file, gt_file, gtDataDir, benchmark_name, nargout=5)
        eng.quit()
        update_dict = results[4]
        self.update_values(update_dict)
