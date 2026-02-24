---
description: "Analyze a repository and generate a technology stack blueprint (dependencies, tooling, patterns) in a chosen format."
agent: "Plan Mode - Strategic Planning & Architecture"
tools: ["codebase", "search", "web/fetch"]
---

# Comprehensive Technology Stack Blueprint Generator

## Preconditions (Strict Spec-First)

Prefer terminology and boundaries from `spec/spec-architecture.md`.

## Inputs

- `${input:PROJECT_TYPE}`: `Auto-detect|.NET|Java|JavaScript|React.js|React Native|Angular|Python|Other` (default: `Auto-detect`)
- `${input:DEPTH_LEVEL}`: `Basic|Standard|Comprehensive|Implementation-Ready` (default: `Standard`)
- `${input:INCLUDE_VERSIONS}`: `true|false` (default: `true`)
- `${input:INCLUDE_LICENSES}`: `true|false` (default: `false`)
- `${input:INCLUDE_DIAGRAMS}`: `true|false` (default: `false`)
- `${input:INCLUDE_USAGE_PATTERNS}`: `true|false` (default: `false`)
- `${input:INCLUDE_CONVENTIONS}`: `true|false` (default: `true`)
- `${input:OUTPUT_FORMAT}`: `Markdown|JSON|YAML|HTML` (default: `Markdown`)
- `${input:CATEGORIZATION}`: `Technology Type|Layer|Purpose` (default: `Technology Type`)

## Workflow

Analyze the codebase and generate a `${input:DEPTH_LEVEL}` technology stack blueprint that documents technologies and implementation patterns. Follow this approach:

### 1. Technology Identification Phase

- If `${input:PROJECT_TYPE}` is `Auto-detect`, scan configuration and code to determine the primary stack(s).
- Otherwise, focus primarily on `${input:PROJECT_TYPE}` and document adjacent technologies actually present.
- Identify all programming languages by examining file extensions and content
- Analyze configuration files (package.json, .csproj, pom.xml, etc.) to extract dependencies
- Examine build scripts and pipeline definitions for tooling information
- If `${input:INCLUDE_VERSIONS}` is `true`, extract precise version information from package/config files
- If `${input:INCLUDE_LICENSES}` is `true`, document license information when it is available in manifests or lock files

### 2. Core Technologies Analysis

For each detected or selected stack area, add a section (only when evidence exists in the repo):

- .NET: target frameworks, NuGet deps, configuration, DI, middleware, data access, API patterns
- Java: JDK/frameworks, Maven/Gradle deps, package organization, DI, data access, API patterns
- JavaScript/TypeScript: runtime/module system, dependencies, build tooling, testing
- React/Angular: framework version (if determinable), patterns, routing, state, UI libs
- Python: Python version (if determinable), deps/venv, frameworks, project structure

### 3. Implementation Patterns & Conventions

If `${input:INCLUDE_CONVENTIONS}` is `true`, document conventions and patterns for each technology area:

#### Naming Conventions

- Class/type naming patterns
- Method/function naming patterns
- Variable naming conventions
- File naming and organization conventions
- Interface/abstract class patterns

#### Code Organization

- File structure and organization
- Folder hierarchy patterns
- Component/module boundaries
- Code separation and responsibility patterns

#### Common Patterns

- Error handling approaches
- Logging patterns
- Configuration access
- Authentication/authorization implementation
- Validation strategies
- Testing patterns

### 4. Usage Examples

If `${input:INCLUDE_USAGE_PATTERNS}` is `true`, extract representative code examples showing standard patterns:

#### API Implementation Examples

- Standard controller/endpoint implementation
- Request DTO pattern
- Response formatting
- Validation approach
- Error handling

#### Data Access Examples

- Repository pattern implementation
- Entity/model definitions
- Query patterns
- Transaction handling

#### Service Layer Examples

- Service class implementation
- Business logic organization
- Cross-cutting concerns integration
- Dependency injection usage

#### UI Component Examples (if applicable)

- Component structure
- State management pattern
- Event handling
- API integration pattern

### 5. Technology Stack Map

If `${input:DEPTH_LEVEL}` is `Comprehensive` or `Implementation-Ready`, create a technology map including:

#### Core Framework Usage

- Primary frameworks and their specific usage in the project
- Framework-specific configurations and customizations
- Extension points and customizations

#### Integration Points

- How different technology components integrate
- Authentication flow between components
- Data flow between frontend and backend
- Third-party service integration patterns

#### Development Tooling

- IDE settings and conventions
- Code analysis tools
- Linters and formatters with configuration
- Build and deployment pipeline
- Testing frameworks and approaches

#### Infrastructure

- Deployment environment details
- Container technologies
- Cloud services utilized
- Monitoring and logging infrastructure

### 6. Technology-Specific Implementation Details

Add technology-specific details for only the stacks detected in the repo. Prefer concrete references to actual configuration/code locations.

### 7. Blueprint for New Code Implementation

If `${input:DEPTH_LEVEL}` is `Implementation-Ready`, add an implementation blueprint:

- **File/Class Templates**: Standard structure for common component types
- **Code Snippets**: Ready-to-use code patterns for common operations
- **Implementation Checklist**: Standard steps for implementing features end-to-end
- **Integration Points**: How to connect new code with existing systems
- **Testing Requirements**: Standard test patterns for different component types
- Documentation requirements for new features

If `${input:INCLUDE_DIAGRAMS}` is `true`, add a section with Mermaid diagrams:

- **Stack Diagram**: Visual representation of the complete technology stack
- **Dependency Flow**: How different technologies interact
- **Component Relationships**: How major components depend on each other
- Data Flow: how data flows through the stack

### 8. Technology Decision Context

- Document apparent reasons for technology choices
- Note any legacy or deprecated technologies marked for replacement
- Identify technology constraints and boundaries
- Document technology upgrade paths and compatibility considerations

Format the output as `${input:OUTPUT_FORMAT}` and categorize technologies by `${input:CATEGORIZATION}`.

## Output Expectations

Create the output file at the repository root using this exact mapping:

- `Markdown` -> `Technology_Stack_Blueprint.md`
- `JSON` -> `Technology_Stack_Blueprint.json`
- `YAML` -> `Technology_Stack_Blueprint.yaml`
- `HTML` -> `Technology_Stack_Blueprint.html`

Only include technologies that are evidenced by files in the repository.
