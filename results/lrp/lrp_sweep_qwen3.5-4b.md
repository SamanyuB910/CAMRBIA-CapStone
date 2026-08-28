# LRP per-rule ablation — qwen3.5-4b

Which of the three LRP rules carries the R-lens's improvement? Each arm is a lens
fitted with one subset of {LN, identity, half} on **identical prompts and recipe**,
so differences are attributable to the rules alone.

## Arms fitted (provenance check)

7 of 8 arms present; config_json matches the arm name for 7/7.

| config      |   n_prompts | fit_commit   | prompt_rows              | ok   |
|:------------|------------:|:-------------|:-------------------------|:-----|
| j           |          25 | aa876ee      | [0, 1, 2, 3, 4, 5, 6, 7, | True |
| ln          |          25 | d6696c3      | [0, 1, 2, 3, 4, 5, 6, 7, | True |
| identity    |          25 | 8c57be0      | [0, 1, 2, 3, 4, 5, 6, 7, | True |
| half        |          25 | e8823e1      | [0, 1, 2, 3, 4, 5, 6, 7, | True |
| ln+identity |          25 | f54aeb6      | [0, 1, 2, 3, 4, 5, 6, 7, | True |
| ln+half     |          25 | f54aeb6      | [0, 1, 2, 3, 4, 5, 6, 7, | True |
| r           |          25 | aa876ee      | [0, 1, 2, 3, 4, 5, 6, 7, | True |

**Note — arms come from 5 different fitting runs** (see `fit_commit`). Rule effects are only cleanly attributable if every arm shares the recipe; check `n_prompts` and `prompt_rows` agree across arms, and treat cross-run comparisons as carrying an extra source of variation (different hardware means different kernels, hence tiny numerical differences).

## pass@10 lift over the J-lens arm

All layers:

| config      |   pass@10 |   lift_over_j |   lift_sem |   rel_lift_% |
|:------------|----------:|--------------:|-----------:|-------------:|
| j           |    0.1023 |        0.0000 |     0.0000 |       0.0000 |
| ln          |    0.0754 |       -0.0269 |     0.0072 |     -26.2789 |
| identity    |    0.1075 |        0.0052 |     0.0023 |       5.1156 |
| half        |    0.1219 |        0.0196 |     0.0061 |      19.1310 |
| ln+identity |    0.1026 |        0.0003 |     0.0047 |       0.3153 |
| ln+half     |    0.1021 |       -0.0002 |     0.0029 |      -0.2102 |
| r           |    0.1202 |        0.0179 |     0.0045 |      17.5193 |

First half of layers (where the post locates the R-lens advantage):

| config      |   pass@10 |   lift_over_j |   lift_sem |   rel_lift_% |
|:------------|----------:|--------------:|-----------:|-------------:|
| j           |    0.0662 |        0.0000 |     0.0000 |       0.0000 |
| ln          |    0.0182 |       -0.0480 |     0.0102 |     -72.4832 |
| identity    |    0.0633 |       -0.0029 |     0.0024 |      -4.3624 |
| half        |    0.1098 |        0.0436 |     0.0089 |      65.7718 |
| ln+identity |    0.0518 |       -0.0144 |     0.0045 |     -21.8121 |
| ln+half     |    0.0707 |        0.0044 |     0.0043 |       6.7114 |
| r           |    0.0962 |        0.0300 |     0.0072 |      45.3020 |

## Rule interactions

`excess > 0`: the rules cooperate. `< 0`: they overlap (fix the same failure).

| pair          |   sum_of_parts |   joint |   excess | verdict     |
|:--------------|---------------:|--------:|---------:|:------------|
| ln+identity   |        -0.0216 |  0.0003 |   0.0220 | cooperative |
| ln+half       |        -0.0073 | -0.0002 |   0.0071 | cooperative |
| r (all three) |        -0.0021 |  0.0179 |   0.0200 | cooperative |

## Weight-space geometry — which rule moves the lens?

Mean per-layer cosine of vec(J_l) to each endpoint. A variant close to `r` in
weight space but without its pass@10 lift means the rule changes the lens
substantially in a direction that does not help the metric.

| config      |   cos_to_j |   cos_to_r |   cos_to_released_j |   cos_to_released_r |
|:------------|-----------:|-----------:|--------------------:|--------------------:|
| j           |     1.0010 |     0.8791 |              0.9752 |              0.8750 |
| ln          |     0.9561 |     0.8788 |              0.9576 |              0.8743 |
| identity    |     0.9538 |     0.8918 |              0.9557 |              0.8885 |
| half        |     0.8910 |     0.9850 |              0.8971 |              0.9811 |
| ln+identity |     0.9086 |     0.8844 |              0.9088 |              0.8803 |
| ln+half     |     0.8858 |     0.9901 |              0.8916 |              0.9858 |
| r           |     0.8791 |     1.0011 |              0.8846 |              0.9971 |