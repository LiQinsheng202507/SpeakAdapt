import jiwer
from sacrebleu.metrics import BLEU

def compute_wer(refs, hyps):
    """
    计算词错误率（WER）
    Args:
        refs: List[str] - 参考文本
        hyps: List[str] - 识别输出
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.SentencesToListOfWords()
    ])
    return jiwer.wer(refs, hyps, truth_transform=transformation, hypothesis_transform=transformation)

def compute_cer(refs, hyps):
    """
    计算字错误率（CER）
    """
    return jiwer.cer(refs, hyps)

def compute_bleu(refs, hyps):
    """
    计算 BLEU 分数（多语言模型可选）
    Args:
        refs: List[str]
        hyps: List[str]
    """
    bleu = BLEU()
    return bleu.corpus_score(hyps, [refs]).score
