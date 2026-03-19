# evaluation.csv schema

Header must be:

`eval_id,exp_id_ref,date,job_id,test_config,status`

Field notes:
- `eval_id`: auto-generated as `eval_XXX` by this skill.
- `exp_id_ref`: required input from user.
- `date`: current date in `%Y-%m-%d`.
- `job_id`: from `--job-id` or `${SLURM_JOB_ID:-unknown}`.
- `test_config`: compact summary string; always quoted.
- `status`: default `submitted`.
