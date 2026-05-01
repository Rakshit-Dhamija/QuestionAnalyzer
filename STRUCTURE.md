# Project Structure

```
QuestionAnalyzer/
├── app/                          # Main application code
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   └── web-app.py                # Main Flask/web application
│
├── data/                         # Data storage
│   ├── raw/                      # Raw input data
│   └── processed/                # Processed data files (CSV, XLSX, TXT)
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── Bert_mode.ipynb
│   ├── data_manipulate.ipynb
│   ├── extractQuestions.ipynb
│   ├── questions_extractor.ipynb
│   └── test.ipynb
│
├── static/                       # Static files
│   ├── css/                      # Stylesheets
│   ├── js/                       # JavaScript files
│   └── images/                   # Images and media
│
├── templates/                    # HTML templates
│
├── scripts/                      # Utility scripts
│
├── tests/                        # Unit and integration tests
│
├── docs/                         # Documentation and references
│   └── Primer.pdf
│
├── requirements.txt              # Python dependencies (pip)
├── Pipfile                       # Python dependencies (pipenv)
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore rules
```

## Folder Purposes

- **app/**: Contains all application code, main entry point, and configuration
- **data/**: Stores all data (raw in `raw/`, processed in `processed/`)
- **notebooks/**: Jupyter notebooks for exploratory data analysis and prototyping
- **static/**: CSS, JavaScript, and image files served to the frontend
- **templates/**: HTML templates for the web application
- **scripts/**: Reusable Python scripts for data processing and utilities
- **tests/**: Unit tests and integration tests
- **docs/**: Documentation, guides, and reference materials

## Next Steps

1. Create your Flask/web routes in `app/routes.py` or similar
2. Add HTML templates to `templates/`
3. Add CSS/JS to `static/css/` and `static/js/`
4. Develop utility functions in `scripts/`
5. Create unit tests in `tests/`
