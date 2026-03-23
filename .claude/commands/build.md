Execute the next phase of the PSWM build. Follow this exact protocol:

1. Read CLAUDE.md and the Build Order section
2. Run `find pswm/ -name "*.py" -exec grep -l "def " {} \;` to see which files have real implementations
3. Run `python -m pytest tests/ -v --tb=short 2>&1 | tail -5` to check current test status
4. Identify the NEXT unbuilt item in the Build Order (a file that either doesn't exist or only has a placeholder)
5. Implement it fully:
   - Follow all conventions in CLAUDE.md
   - Type hints on everything
   - Google-style docstrings
   - Handle both batched and unbatched tensors where applicable
6. Write tests in the corresponding test file
7. Run tests and fix any failures
8. Run ruff and black to format
9. Stage, commit with a descriptive message, and push
10. Report what you built and what's next in the Build Order

If ALL items in the Build Order are complete, say "BUILD ORDER COMPLETE" and suggest what to work on next.