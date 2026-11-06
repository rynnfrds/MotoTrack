# Aturan Kerja di Repo Ini

## Atribusi commit — WAJIB

Setiap commit dan push HARUS atas nama **rynnfrds <rynstudypurpose@gmail.com>** saja.

- JANGAN pernah menambahkan `Co-Authored-By:` (agent, model, tool apa pun).
- JANGAN menambahkan trailer `Generated-By`, `Assisted-By`, `Signed-off-by` AI, atau sejenisnya.
- JANGAN menambahkan footer "Generated with Claude Code" / "Made with OpenCode" / sejenisnya.
- JANGAN mengubah `user.name` / `user.email` git, dan jangan commit dengan `--author` lain.
- Pesan commit hanya berisi Conventional Commits + body penjelasan perubahan. Tidak ada atribusi.

Tujuan: kontributor di GitHub hanya akun manusia (owner repo), bukan agent.

## Bahasa

- Balasan ke user: Bahasa Indonesia, ringkas.
- Pesan commit: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`),
  body menjelaskan sebab-akibat perubahan, bukan sekadar daftar file.
