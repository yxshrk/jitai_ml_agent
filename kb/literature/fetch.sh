#!/usr/bin/env bash
# Re-download the papers listed in README.md (PDFs are not committed; run this once after cloning).
set -u; cd "$(dirname "$0")"
dl() { mkdir -p "$1"; [ -f "$1/$2.pdf" ] && { echo "have  $1/$2.pdf"; return; }
      curl -sL --max-time 90 -A "Mozilla/5.0" -o "$1/$2.pdf" "$3" && head -c 5 "$1/$2.pdf" | grep -q '%PDF' \
      && echo "OK    $1/$2.pdf" || { echo "FAIL  $1/$2.pdf  $3"; rm -f "$1/$2.pdf"; }; }
dl agents    2410.07095_mle-bench                     https://arxiv.org/pdf/2410.07095
dl agents    2502.13138_aide                          https://arxiv.org/pdf/2502.13138
dl agents    2504.08066_ai-scientist-v2               https://arxiv.org/pdf/2504.08066
dl dataset   2208.08696_kuairand                      https://arxiv.org/pdf/2208.08696
dl models    rendle2010_factorization-machines        https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf
dl models    1703.04247_deepfm                        https://arxiv.org/pdf/1703.04247
dl models    1708.05123_dcn                           https://arxiv.org/pdf/1708.05123
dl models    2008.13535_dcn-v2                        https://arxiv.org/pdf/2008.13535
dl models    1803.05170_xdeepfm                       https://arxiv.org/pdf/1803.05170
dl models    1706.06978_din                           https://arxiv.org/pdf/1706.06978
dl models    2006.05639_sim                           https://arxiv.org/pdf/2006.05639
dl losses    1205.2618_bpr                            https://arxiv.org/pdf/1205.2618
dl losses    burges2010_ranknet-lambdarank-lambdamart https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf
dl losses    cao2007_listnet                          https://www.microsoft.com/en-us/research/uploads/prod/2016/02/tr-2007-40.pdf
dl multitask 1804.07931_esmm                          https://arxiv.org/pdf/1804.07931
dl watchtime 2406.07932_cwm                           https://arxiv.org/pdf/2406.07932
dl watchtime 2206.06003_d2q                           https://arxiv.org/pdf/2206.06003
dl watchtime 2306.03392_tpm                           https://arxiv.org/pdf/2306.03392
dl watchtime 2308.08120_biased-noised-watchtime       https://arxiv.org/pdf/2308.08120
