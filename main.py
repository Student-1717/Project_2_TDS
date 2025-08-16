@app.post("/api/")
async def analyze(request: Request,
                  questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None,
                  yaml_file: Optional[UploadFile] = File(None)):

    try:
        # --- Step 1: Read questions ---
        questions_content = ""
        body = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            questions_content = body.get("request", "").strip()
        elif questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        questions, urls = parse_questions(questions_content)

        # --- Step 2: Process uploaded files ---
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                else:
                    uploaded_data[f.filename] = content

        # --- Step 3: Scrape URLs ---
        dataframes = {}
        for url in urls:
            df = scrape_table_from_url(url)
            dataframes[url] = df
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # --- Step 4: Load YAML dynamically ---
        parsed_yaml = {}
        if yaml_file:
            yaml_content = (await yaml_file.read()).decode("utf-8")
            parsed_yaml = yaml.safe_load(yaml_content)

        # --- Step 5: Precompute all YAML metrics ---
        precomputed_metrics = {}
        chart_cache = {}
        evaluations = parsed_yaml.get("evaluations", [])
        for eval_config in evaluations:
            csv_file = eval_config.get("csv_file")
            metrics = eval_config.get("metrics", {})
            eval_name = eval_config.get("name", csv_file)

            if csv_file not in dataframes:
                continue
            df = dataframes[csv_file]

            # Compute metrics once
            computed_metrics = {}
            for key, props in metrics.items():
                col = props.get("column")
                op = props.get("operation")
                if col not in df.columns:
                    computed_metrics[key] = None
                    continue
                if op == "sum":
                    computed_metrics[key] = df[col].sum()
                elif op == "mean":
                    computed_metrics[key] = df[col].mean()
                elif op == "median":
                    computed_metrics[key] = df[col].median()
                elif op == "max":
                    computed_metrics[key] = df[col].max()
                elif op == "min":
                    computed_metrics[key] = df[col].min()
                elif op == "count":
                    computed_metrics[key] = df[col].count()
                else:
                    computed_metrics[key] = None
            precomputed_metrics.update(computed_metrics)

            # Generate charts once per numeric column
            numeric_cols = df.select_dtypes(include="number").columns
            for col in numeric_cols:
                chart_cache[f"{eval_name}_{col}_chart"] = generate_scatterplot(df, col, col)

        # --- Step 6: Map questions to precomputed metrics ---
        answers_dict = {}
        for question in questions:
            matched = False
            for key, value in precomputed_metrics.items():
                if key.lower() in question.lower():
                    answers_dict[question] = value
                    matched = True
                    break
            if not matched:
                # Call AI only if no precomputed match
                value = await ai_generate_value_for_key("unknown", question, dataframes)
                answers_dict[question] = value

        # --- Step 7: Add generated charts ---
        answers_dict.update(chart_cache)

        return JSONResponse({"dict": answers_dict, "array": list(answers_dict.values())})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
