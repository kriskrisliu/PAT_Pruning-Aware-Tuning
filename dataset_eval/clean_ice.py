import json, os

for _file in os.listdir("./"):
    # if _file == "lukaemon_mmlu_electrical_engineering.json":
    #     continue
    if not _file.endswith(".json"):
        continue
    with open(_file,"r") as fp:
        ctx = json.load(fp)
    for key,val in ctx.items():
        origin_prompt = val["origin_prompt"]
        new_prompt = origin_prompt.split("\nAnswer: \n")[-1]
        new_prompt = new_prompt[3:]
        val["origin_prompt"] = new_prompt
    with open(_file,"w") as fp:
        json.dump(ctx, fp, indent=4)