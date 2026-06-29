---
description: "Documentation and content creation standards"
applyTo: "{README.md,CHANGELOG.md,CONTRIBUTING.md,docs/**/*.md,**/design.md}"
---

## Content Rules

1. **Headings**: Use `##` for H2 and `###` for H3 in hierarchical order. Avoid H1
   unless the document type requires it (e.g., README). Restructure if content
   reaches H4; avoid H5.
2. **Lists**: Use `-` for bullets and `1.` for numbered lists. Indent nested lists
   with two spaces.
3. **Code Blocks**: Use fenced blocks with a language specifier for syntax
   highlighting (e.g., \`\`\`python).
4. **Links**: Use `[descriptive text](URL)` with valid, accessible URLs.
5. **Images**: Use `![alt text](URL)` with a brief description in the alt text.
6. **Tables**: Use `|` columns with aligned headers.
7. **Line Length**: Wrap prose around 80 characters; 400 characters absolute max.
8. **Whitespace**: Use blank lines to separate sections; avoid excessive whitespace.
9. **Front Matter**: Optional. Require YAML front matter only for blog-style posts
   when the repository defines a front-matter schema.
