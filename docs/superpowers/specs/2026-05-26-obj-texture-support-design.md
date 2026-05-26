# OBJ + textura: extensão do pipeline e do viewer

**Data:** 2026-05-26
**Autor:** Claude (autônomo — Vinicius autorizou merge a `main` em ambos os repos se o resultado passar nos testes)

## Resumo

Hoje o pipeline aceita apenas STL e gera GLB multi-mesh com cores PBR por keyword.
Estender pra aceitar OBJ + MTL + imagem de textura (de aplicativos como KIRI Engine, photogrammetry) preservando a textura embedada no GLB. Viewer já renderiza textura nativamente via `GLTFLoader` — não precisa de mudança estrutural lá, só ajustes pontuais.

Trabalho cobre dois repos:

- `mesh-processor` (backend FastAPI): aceitar OBJ bundle, conversão, scale m→mm, GLB com textura
- `medCaseViewer/washington` (frontend estático): página de upload aceitar OBJ; viewer permanece igual

## Decisões (documentadas, não negociadas — usuário está dormindo)

### Entrada

- **Formatos aceitos no `/upload`**, detectados por extensão:
  1. **OBJ bundle multipart**: 1 `.obj` + 1 `.mtl` + 1+ imagens (`.jpg`/`.jpeg`/`.png`) enviados como múltiplos `files` no `FormData` — exatamente como o `.zip` que o KIRI exporta, só que descompactado.
  2. **OBJ bundle como `.zip`**: 1 `.zip` contendo os 3 arquivos. Servidor descompacta em memória.
  3. **STL** (comportamento atual): 1+ arquivos `.stl`. Preservado intacto.
- **Mistura proibida**: ou tudo é STL, ou é um único OBJ bundle. `HTTP 400` com mensagem em pt-BR se vier misturado.
- **Múltiplos OBJ no mesmo caso**: não suportado em v1. KIRI exporta um modelo de cada vez; suporte a multi-OBJ exige protocolo pra mapear cada `.obj` ao seu `.mtl` + textura. YAGNI.
- **UI do upload** menciona "STL" ou "OBJ". `.zip` fica no `accept=` do input silenciosamente — o usuário disse explicitamente que falar de `.zip` no texto fica estranho.
- **Cap de 60MB** preservado.

### Processamento

- **Sem decimação para OBJ.** `fast_simplification` não preserva UVs — decimar quebraria o mapping da textura. O cap de 60MB no upload já limita o tamanho. STL continua decimando como antes.
- **Conversão de unidades**: vértices do OBJ são multiplicados por **1000** (metros → milímetros). O viewer assume 1 unidade Three.js = 1 mm em todo lugar (medição, AR, escala USDZ). Aplicativos tipo KIRI exportam em metros por convenção.
- **Sem rotação RAS→Y-up.** STL vem como RAS (Z-up) de software de segmentação médica. OBJ photogrammetry já vem Y-up. Verificado no fixture: bounds são `[[-0.35, 0, -0.32], [0.35, 0.85, 0.32]]` — Y é o eixo "alto".
- **Cor**: usuário disse explicitamente "use a textura, não as cores". O `process_obj` não roda a lógica de keyword/HSV-bucket — só carrega o material que veio com o OBJ, deixa o `Scene.export(glb)` converter `SimpleMaterial` → `PBRMaterial` com `baseColorTexture`. Verificado: trimesh faz essa conversão automática.
- **Multi-mesh dentro de um OBJ** (vários `o ...`): se trimesh devolver `Scene`, concatena num único `Trimesh` mantendo a textura. Photogrammetry é uma estrutura única na prática.
- **Nome da mesh**: filename do `.obj` (sem extensão), passado pelo mesmo `clean_mesh_names` que limpa STLs.

### Saída

- Mesma resposta do `/upload` atual (`uid`, `viewer_url`, `stats`). Campo `meshes[].color` vira `null` quando textura presente — o viewer não exibe swatch quando a cor não vem.
- Sketchfab + R2 upload idênticos. Sketchfab também aceita GLB com textura PBR (`isDownloadable=true` mantém bypass do cap mensal).

### Viewer (`medCaseViewer/case/`)

- **Não mexer no render core.** `world.js > mount()` já clona `child.material` e força `transparent: true; depthWrite: true`. Para um material com `baseColorTexture` opaca, isso é no-op visualmente.
- **Painel "Estruturas"**: hoje exibe um swatch colorido. Pra mesh com textura, `getMeshColor()` retorna `#FFFFFF` (default do PBR baseColorFactor) — vai parecer um swatch branco. Aceitável. Se virar UX issue, próximo PR.
- **Medição/AR**: dependem só de `mesh.geometry.attributes.position` + `matrixWorld`. Textura não muda nada. O scale mm→m do AR USDZ continua valendo.

### Arquitetura (mesh-processor)

```
main.py
  └─ /upload route
       ├─ sniff extensions
       ├─ if all .stl  → processor.process_stls(...)
       ├─ if .obj+.mtl+img OR .zip → processor.process_obj_bundle(...)
       └─ else → HTTPException(400)

processor.py (pure, no I/O)
  ├─ process_stls(...)           [unchanged]
  └─ process_obj_bundle(obj_bytes, mtl_bytes, textures, name)
       ├─ write to temp dir (trimesh resolves MTL/JPG relative to OBJ path)
       ├─ load → Trimesh (concatena se Scene)
       ├─ apply_scale(1000)
       ├─ force vertex_normals compute
       └─ scene.add_geometry + scene.export("glb")
```

`process_obj_bundle` é uma função paralela a `process_stls`. Compartilham `ProcessStats` / `MeshStats` mas não há refactor de base — não vale o churn por uma função.

### Tratamento de erros

- ZIP com mais de 1 OBJ: `400 "Bundle OBJ deve conter exatamente um arquivo .obj."`
- ZIP sem MTL ou textura: `400 "Bundle OBJ precisa do .mtl e ao menos uma imagem de textura."`
- Mistura STL+OBJ na mesma request: `400 "Envie apenas arquivos STL ou um único bundle OBJ."`
- OBJ com geometria vazia: `400 "OBJ inválido ou corrompido: ..."` (reaproveita o pattern do STL)

### Testes

1. **`test_processor.py`**: adicionar `test_obj_bundle()` usando o fixture do attachment.
   - Verificar: GLB válido, textura embedada, vértices escalados (~700mm × 850mm × 640mm).
2. **Manual e2e**: rodar `uvicorn` localmente com `DRY_RUN=true`, fazer curl com OBJ bundle, abrir o GLB resultante no viewer Three.js (servido localmente) — confirmar render com textura.
3. **STL não-regressão**: rodar o `test_processor.py` existente sem modificações.

### Critérios de "confiança pra merge"

Antes de mergeio nos `main`:

- [ ] Smoke test STL existente passa
- [ ] Novo test OBJ passa
- [ ] `uvicorn` boota com `DRY_RUN=true`
- [ ] `/upload` aceita o fixture OBJ → GLB → resposta com `uid`
- [ ] GLB resultante carrega no `case/index.html` local com textura visível
- [ ] Medição linear no GLB com textura retorna número plausível (espera-se ~mm em modelo de ~85cm = ~850mm)
- [ ] `dom.renderStructures` não crasha sem cor
- [ ] AR button monta (mesmo sem dispositivo real pra testar — só checar que ar.init não joga erro)

Se algo falhar e eu não conseguir resolver com confiança, deixo no branch e documento no resumo final.
