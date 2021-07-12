import train_so2sat
import test_so2sat
from benchmarking import bm
import e2ebench as eb


tta_tracker = eb.TTATracker(bm)
cmx_tracker = eb.ConfusionMatrixTracker(bm)


def main():
    train_result = train_so2sat.train()
    tta_tracker.track(train_result["accuracy"], "train time to accuracy")

    test_result = test_so2sat.test(train_result["model"])
    cmx_tracker.track(test_result["confusion matrix"], test_result["classes"], "confusion matrix")

    bm.close()


if __name__ == "__main__":
    main()
