# Resume Scanner Usage Guide

## How the Resume Scanner Works

The app intelligently extracts information from your resume using pattern matching and NLP techniques.

### What Gets Extracted

1. **CGPA/GPA** (High Confidence)
   - Patterns: "CGPA: 8.5", "GPA 3.5/4.0", "8.5 CGPA"
   - Also converts percentages to 10-point scale
   - Normalizes 4.0 scale to 10-point scale

2. **Projects** (High Confidence)
   - Counts numbered projects in PROJECTS section
   - Looks for keywords: "project", "capstone", "thesis"
   - Detects numbered lists (1., 2., etc.)

3. **Internship Experience** (High Confidence)
   - Keywords: "intern", "internship", "trainee", "co-op", "work experience"
   - Binary: Yes/No

4. **Extra-Curricular Activities** (Medium Confidence)
   - Keywords: "volunteer", "club", "sport", "leadership", "award"
   - Also: "certificate", "achievement", "competition", "hackathon", "event"
   - Counts occurrences

5. **Communication Skills** (Medium Confidence)
   - Estimated from resume quality:
     - Word count (300+ words = better score)
     - Well-structured sections
     - Presence of summary/objective

6. **Academic Performance** (Medium Confidence)
   - Inferred from CGPA:
     - CGPA >= 8.5 → Academic Performance = 9
     - CGPA >= 7.5 → Academic Performance = 8
     - CGPA >= 6.5 → Academic Performance = 7

7. **IQ** (Assumed - Default)
   - Set to 100 (cannot be extracted from resume)
   - You should adjust this manually if known

### Resume Format Tips

For best extraction results:

1. **Use clear section headings**
   ```
   EDUCATION
   EXPERIENCE
   PROJECTS
   ACHIEVEMENTS
   ```

2. **Number your projects**
   ```
   PROJECTS
   1. Project Name - Description
   2. Project Name - Description
   ```

3. **Include CGPA/GPA clearly**
   ```
   CGPA: 8.7/10.0
   or
   GPA: 3.5/4.0
   ```

4. **List internships in EXPERIENCE section**
   ```
   Software Engineering Intern
   Company Name, Duration
   ```

5. **Mention extracurricular activities**
   ```
   ACHIEVEMENTS
   - Winner of Hackathon 2023
   - Volunteer at NGO
   - Club President
   ```

### After Upload

1. Review the **Extracted Information** panel
2. Check the **Findings** - what was detected
3. Verify the **Extracted Values** metrics
4. **Adjust sliders below** if any value seems incorrect
5. Click **Predict Placement**

### Supported Formats

- **PDF** (requires PyPDF2 installed)
- **TXT** (plain text)

### Example

See `sample_resume.txt` in the project folder for a reference resume format.
