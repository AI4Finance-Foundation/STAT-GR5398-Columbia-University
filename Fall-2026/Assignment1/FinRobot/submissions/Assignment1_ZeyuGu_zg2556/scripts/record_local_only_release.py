"""Apply the user's explicit local-only direction to current release documents."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
replacements={
    'not yet available: awaiting final release authorization':'not published: user requested local-only materials',
    'Medium publication and the course master pull request are pending final release authorization.':'The user has explicitly deferred Medium publication and the course master pull request; the materials remain local.',
    'They remain pending, so this document does not declare the entire assignment formally submitted.':'Publication is deferred at the user’s request, so this document does not declare the entire assignment formally submitted.',
    'Await final user publication authorization and authenticated Medium account if needed.':'Publication is explicitly deferred by the user; keep the complete article local and do not request authorization again unless the user resumes submission.',
    'Await final user course-master PR authorization; use isolated checkout/fork without switching/resetting shared worktree or replacing folder with a branch.':'Course PR is explicitly deferred by the user; keep the scoped local diff. If the user later resumes submission, use an isolated checkout/fork without switching/resetting the shared worktree.',
    "'Final Medium publication authorization and actual URL','Final course master PR authorization and actual PR'":"'Medium publication and real URL deferred by explicit user direction','Course master PR deferred by explicit user direction'",
    'Publish authorized article and insert genuine URL after success.':'Keep local per current user direction; only insert a genuine URL after a later authorized publication succeeds.',
    'Final Medium and course-master PR submission require the user\'s authorization of these reviewable materials; no real URL will be invented.':'The user has explicitly requested that Medium and course-master PR publication be deferred. Keep these materials local until the user resumes submission; no public URL is invented.',
}
for name in ['README.md','blog/medium_article.md','docs/research_report.md',
    'scripts/write_release_docs.py','scripts/finalize_research_materials.py','scripts/checkpoint_release.py']:
    path=ROOT/name;text=path.read_text(encoding='utf-8')
    for before,after in replacements.items():text=text.replace(before,after)
    path.write_text(text,encoding='utf-8')
print('Recorded explicit local-only publication direction in release sources.')
