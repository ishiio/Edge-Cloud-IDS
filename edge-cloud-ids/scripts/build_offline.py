"""Create the portable HTML with bundled real data previews and saved results."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
web=root/'web'
s=(web/'index.html').read_text()
s=s.replace('<link rel="stylesheet" href="/static/app.css">', '<style>'+(web/'app.css').read_text()+'</style>')
for name in ['snapshot.js','app.js','datasets.js']:
    s=s.replace('<script src="/static/'+name+'"></script>', '<script>'+(web/name).read_text().replace('</script','<\\/script')+'</script>')
(root/'OPEN_DASHBOARD.html').write_text(s)
print('Portable dashboard written.')
