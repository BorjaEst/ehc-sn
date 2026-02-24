---
description: "Analyze repository folder structure and generate a folder-structure blueprint document."
agent: "Plan Mode - Strategic Planning & Architecture"
tools: ["codebase", "search"]
---

# Project Folder Structure Blueprint Generator

## Preconditions (Strict Spec-First)

Describe the repo using the architecture taxonomy from `spec/spec-architecture.md`.

## Inputs

- `${input:PROJECT_TYPE}`: `Auto-detect|.NET|Java|React|Angular|Python|Node.js|Flutter|Other` (default: `Auto-detect`)
- `${input:INCLUDES_MICROSERVICES}`: `Auto-detect|true|false` (default: `Auto-detect`)
- `${input:INCLUDES_FRONTEND}`: `Auto-detect|true|false` (default: `Auto-detect`)
- `${input:IS_MONOREPO}`: `Auto-detect|true|false` (default: `Auto-detect`)
- `${input:VISUALIZATION_STYLE}`: `ASCII|Markdown List|Table` (default: `ASCII`)
- `${input:DEPTH_LEVEL}`: `1-5` (default: `3`)
- `${input:INCLUDE_FILE_COUNTS}`: `true|false` (default: `false`)
- `${input:INCLUDE_GENERATED_FOLDERS}`: `true|false` (default: `false`)
- `${input:INCLUDE_FILE_PATTERNS}`: `true|false` (default: `true`)
- `${input:INCLUDE_TEMPLATES}`: `true|false` (default: `false`)

## Workflow

### 0) Auto-detection (when requested)

- If `${input:PROJECT_TYPE}` is `Auto-detect`, infer the primary stack by scanning for common markers:
  - .NET: `.sln`, `.csproj`
  - Java: `pom.xml`, `build.gradle`
  - Node: `package.json`
  - Python: `requirements.txt`, `pyproject.toml`

- If `${input:IS_MONOREPO}` is `Auto-detect`, check for multiple distinct subprojects with their own build manifests.
- If `${input:INCLUDES_MICROSERVICES}` is `Auto-detect`, look for multiple service roots (repeating Docker/build/deploy patterns).
- If `${input:INCLUDES_FRONTEND}` is `Auto-detect`, look for UI build config and common frontend folders.

### 1) Structural Overview

Provide a high-level overview of the detected or selected project structure:

- Document the overall architectural approach reflected in the folder structure
- Identify the main organizational principles (by feature, by layer, by domain, etc.)
- Note any structural patterns that repeat throughout the codebase
- Document the rationale behind the structure where it can be inferred

If the repository is a monorepo, explain relationships between subprojects.

If microservices are present, describe how services are separated and composed.

### 2) Directory Visualization

Render the hierarchy to depth `${input:DEPTH_LEVEL}` using `${input:VISUALIZATION_STYLE}`.

- If `${input:INCLUDE_GENERATED_FOLDERS}` is `false`, exclude common generated folders such as `node_modules/`, `dist/`, `build/`, `.venv/`, `__pycache__/`.

### 3) Key Directory Analysis

Document each significant directory's purpose, contents, and patterns:

For the detected/selected technology, describe each significant directory:

- Purpose
- Typical contents
- Conventions

Include technology-specific analysis ONLY when the repo evidence supports it (or when
`${input:PROJECT_TYPE}` explicitly requests it).

#### .NET Project Structure (if detected)

- **Solution Organization**: how projects are grouped and related
- **Project Organization**: internal folder structure patterns and dependencies
- **Domain/Feature Organization**: how domains/features are separated
- **Configuration Management**: where configuration files live and how environments differ
- **Testing Organization**: where tests and test utilities live

#### UI Project Structure (if detected)

- **Component Organization**: grouping strategies and shared vs. feature components
- **State Management**: store structure and state-related file placement
- **Routing Organization**: route definitions and page/view structure
- **API Integration**: API client/service placement and data fetching patterns
- **Assets & Styles**: static resources and CSS/SCSS organization

### 4) File Placement Patterns

If `${input:INCLUDE_FILE_PATTERNS}` is `true`, document the rules that determine where different file types go:

- **Configuration Files**:
  - Locations for different types of configuration
  - Environment-specific configuration patterns
- **Model/Entity Definitions**:
  - Where domain models are defined
  - Data transfer object (DTO) locations
  - Schema definition locations
- **Business Logic**:
  - Service implementation locations
  - Business rule organization
  - Utility and helper function placement
- **Interface Definitions**:
  - Where interfaces and abstractions are defined
  - How interfaces are grouped and organized
- **Test Files**:
  - Unit test location patterns
  - Integration test placement
  - Test utility and mock locations
  - API documentation placement
  - Internal documentation organization
  - README file distribution

### 5) Naming and Organization Conventions

Document the naming and organizational conventions observed across the project:

- **File Naming Patterns**:
  - Case conventions (PascalCase, camelCase, kebab-case)
  - Prefix and suffix patterns
  - Type indicators in filenames
- **Folder Naming Patterns**:
  - Naming conventions for different folder types
  - Hierarchical naming patterns
  - Grouping and categorization conventions
- **Namespace/Module Patterns**:
  - How namespaces/modules map to folder structure
  - Import/using statement organization
  - Internal vs. public API separation

- **Organizational Patterns**:
  - Code co-location strategies
  - Feature encapsulation approaches
  - Cross-cutting concern organization

### 6) Navigation and Development Workflow

Provide guidance for navigating and working with the codebase structure:

- **Entry Points**:
  - Main application entry points
  - Key configuration starting points
  - Initial files for understanding the project

- **Common Development Tasks**:
  - Where to add new features
  - How to extend existing functionality
  - Where to place new tests
  - Configuration modification locations
- **Dependency Patterns**:
  - How dependencies flow between folders
  - Import/reference patterns
  - Dependency injection registration locations

If `${input:INCLUDE_FILE_COUNTS}` is `true`, add a short section with file counts per major directory.

### 7) Build and Output Organization

Document the build process and output organization:

- **Build Configuration**:
  - Build script locations and purposes
  - Build pipeline organization
  - Build task definitions
- **Output Structure**:
  - Compiled/built output locations
  - Output organization patterns
  - Distribution package structure
- **Environment-Specific Builds**:
  - Development vs. production differences
  - Environment configuration strategies
  - Build variant organization

### 8) Technology-Specific Organization

Include the following sections ONLY when that technology is detected in the repo (or when
`${input:PROJECT_TYPE}` is explicitly set to that technology).

#### .NET-Specific Structure Patterns (if detected)

- Project file organization (framework targets, item groups, build props)
- Package management patterns (NuGet config and versioning)

#### Java-Specific Structure Patterns (if detected)

- Package hierarchy and module boundaries
- Build tool organization (Maven/Gradle)

#### Node.js-Specific Structure Patterns (if detected)

- Module organization (CJS vs ESM) and scripts layout
- Configuration management patterns

### 9) Extension and Evolution

Document how the project structure is designed to be extended:

- **Extension Points**:
  - How to add new modules/features while maintaining conventions
  - Plugin/extension folder patterns
  - Customization directory structures
- **Scalability Patterns**:
  - How the structure scales for larger features
  - Approach for breaking down large modules
  - Code splitting strategies
- **Refactoring Patterns**:
  - Common refactoring approaches observed
  - How structural changes are managed
  - Incremental reorganization patterns

If `${input:INCLUDE_TEMPLATES}` is `true`, add templates for extending the structure:

Provide templates for creating new components that follow project conventions:

- **New Feature Template**:
  - Folder structure for adding a complete feature
  - Required file types and their locations
  - Naming patterns to follow
- **New Component Template**:
  - Directory structure for a typical component
  - Essential files to include
  - Integration points with existing structure
- **New Service Template**:
  - Structure for adding a new service
  - Interface and implementation placement
  - Configuration and registration patterns

### 10) Structure Enforcement

Document how the project structure is maintained and enforced:

- **Structure Validation**:
  - Tools/scripts that enforce structure
  - Build checks for structural compliance
  - Linting rules related to structure
- **Documentation Practices**:
  - How structural changes are documented
  - Where architectural decisions are recorded
  - Structure evolution history

Include a section at the end about maintaining this blueprint and when it was last updated.

## Output Expectations

Create `Project_Folders_Structure_Blueprint.md` at the repository root.

- Ensure all referenced paths exist.
- Keep the document concise and navigable.
