# OBJ multiobjeto com material/cor por objeto

**Data:** 2026-06-03
**Autor:** Claude + Vinicius (design fechado em diálogo)

## Resumo

Generalizar o caminho OBJ do `mesh-processor` para aceitar **OBJ multiobjeto**, preservando o **material de cada objeto** (cor chapada e/ou textura) e caindo na **paleta por keyword** (pele etc.) só quando um objeto não tem material nenhum.

**Princípio do usuário:** config de cor mora **num lugar só** (o processor). Nada de cor no viewer. A regra é dirigida pelo arquivo: trouxe material → respeita; não trouxe → o processor pinta.

Não cobre o viewer — ele já renderiza cor/textura por material nativamente via `GLTFLoader`, e o painel de Estruturas já lê `mesh.material`.

### Nota importante (caso `62b0…`)

O caso que motivou isso (`62b0f21d…`) **não saiu deste pipeline** — é um GLB autorado externamente (5 malhas, cores chapadas, `Metal` com `metallicFactor=1`). O código atual não consegue produzir isso (força `metallic=0` no STL, rejeita OBJ multiobjeto). **Esta mudança vale só pra uploads futuros**; o `62b0` já está gravado na R2 e não é afetado.

## A regra (por objeto)

| O objeto tem… | Ação |
|---|---|
| Textura (`map_Kd` resolvida) | preserva a textura (comportamento atual de 1-objeto) |
| Material sem textura (MTL com `Kd`) | preserva a cor → `baseColorFactor` (dielétrico) |
| Nada | `_name_based_material(nome)` — paleta por keyword **+ acabamento por nome** (vide abaixo) |

**Reconciliação com `b400060` (na `main`):** a produção já tem coloração por nome com `pele` = `#FFD09C` e nome contendo `metal` → acabamento metálico (`metallicFactor=1.0`, `roughnessFactor=0.25`, base `#C0C4C8`). Em vez de duplicar, extraí `_name_based_material(name, bucket_counts, fallback_idx)` — **fonte única** usada por `process_stls` **e** pelo caminho "sem material" do OBJ. Assim um objeto OBJ sem material nomeado `metal` também sai metálico.

**Precedência:** material do arquivo **vence** o acabamento por nome. Um objeto com cor de MTL nomeado `metal` mantém a cor (dielétrico); só objetos **sem material** recebem o acabamento metálico por nome.

STL: **inalterado** (sempre sem material → `_name_based_material`).

## Roteamento (`main.py`)

`_extract_obj_bundle` hoje exige `1 obj + 1 mtl + ≥1 textura`. Relaxar:

- Exatamente **1** `.obj` (mantém).
- **MTL opcional** (0 ou 1; >1 → 400).
- **Texturas opcionais** (0+).
- Retorno passa a ser `(obj_bytes, mtl_bytes|None, textures, obj_name)`.
- STL+OBJ misturado: continua **400**.
- Detecção de "é bundle OBJ": presença de `.obj` ou `.zip` (mantém).

## `process_obj_bundle` (reescrita)

Assinatura: `process_obj_bundle(obj_bytes, mtl_bytes: bytes | None, textures, name)`.

1. Materializa em tmpdir: `model.obj` (+ `model.mtl` reescrito no `mtllib` **se houver MTL**) + texturas (basename sanitizado).
2. `trimesh.load(obj_path, process=True)` → `Scene` (multiobjeto) ou `Trimesh` (1 objeto). Normaliza pra lista de `(part_name, Trimesh)`.
3. **Sem rejeição de multiobjeto.** Itera sobre todas as partes.
4. **Escala (heurística de unidade):** computa a maior dimensão do bounding-box do **conjunto** das partes. Se `max_extent < UNIT_HEURISTIC_THRESHOLD` (≈ 10) → assume metros → `apply_scale(1000)` em todas as partes com o **mesmo** fator; senão assume mm e não escala. (Hoje é `×1000` fixo, que estouraria um OBJ médico em mm.) Aplica o mesmo fator a todas as partes pra não desalinhar o conjunto.
5. **Por parte**, decide o material:
   - tem `material.image` (textura) → mantém `visual` como veio.
   - senão, tem cor própria de MTL (`Kd`/`baseColorFactor` setado pelo material) → mantém (export converte pra `baseColorFactor`).
   - senão → atribui `PBRMaterial(baseColorFactor=keyword, metallic=0, roughness=0.5)` via `_pick_color`/`_vary_hsv`, reusando o mesmo `bucket_counts`/`fallback_idx` do STL pra variação HSV consistente.
6. Força `vertex_normals` por parte (após a escala) pra emitir `NORMAL`.
7. `scene.add_geometry(part, node_name=part_name, geom_name=part_name)` por parte → `scene.export("glb")`.

### Nomes das partes
- Multiobjeto: usa a chave do `scene.geometry` (nome do `o`/`g` ou do material), passada por uma limpeza leve (transliterar acentos, strip). Colisões recebem sufixo `_2`, `_3`.
- 1 objeto: usa `name` (filename limpo), como hoje.

### Guarda de textura perdida (Pillow ausente)
Trocar o `raise if material.image is None` (que agora barraria casos legítimos sem textura) por uma guarda cirúrgica: **só falha se o MTL referenciar `map_Kd` mas a imagem correspondente não resolver**. Implementação: escanear o MTL por linhas `map_Kd <arquivo>`; se houver e a parte que usa esse material vier sem `image`, `raise ValueError(...)`. Sem MTL ou sem `map_Kd` → nunca falha por textura.

## Limitações registradas
- **MTL não carrega metalness PBR** (só `Kd`/`Ks`/`Ns`). Então a forma de ter metal num upload é deixar o objeto/STL **sem material** e nomeá-lo `metal` → o `_name_based_material` aplica `metallicFactor=1`. Um objeto OBJ **com** cor de MTL fica dielétrico (a cor do arquivo vence).
- **Sem decimação no OBJ** (quebraria UV) — mantém. Cap de 60MB segura o tamanho.
- **Sem rotação RAS→Y-up no OBJ** — mantém (OBJ já vem Y-up).

## Testes (`pytest`, self-contained — sintetizam OBJ/MTL/PNG em memória, sem fixtures externas)

1. **Multiobjeto, materiais mistos:** OBJ com 2 objetos — `cubeA` com `Kd` (cor chapada), `cubeB` sem material → GLB com 2 malhas nomeadas; A com `baseColorFactor` ≈ Kd, B com cor de keyword/fallback.
2. **Objeto sem material → keyword:** objeto nomeado `pele_*` sem MTL → `baseColorFactor` ≈ `#C4908E`.
3. **Cor chapada preservada:** Kd específico sobrevive ao round-trip (load do GLB exportado).
4. **Textura preservada (não-regressão):** OBJ 1-objeto com `map_Kd` (PNG sintético) → `baseColorTexture` presente.
5. **Heurística de escala:** objeto ~0.3 unidade (metros) → vira ~300mm; objeto ~300 unidades (mm) → fica ~300mm (não estoura pra 300000).
6. **Guarda de textura:** MTL com `map_Kd` apontando imagem ausente → `ValueError`; MTL com só `Kd` (sem `map_Kd`) e imagem ausente → **não** falha.
7. **Roteamento:** `_extract_obj_bundle` aceita OBJ sem MTL e OBJ sem textura; rejeita 2 OBJ e STL+OBJ misturado.
8. **STL não-regressão:** smoke do `process_stls` (cores por keyword) intacto.

## Critérios de pronto
- [ ] Todos os testes acima verdes
- [ ] `process_obj_bundle` com fixture KIRI antigo (textura 1-objeto) ainda passa
- [ ] `uvicorn` boota com `DRY_RUN=true`; `_extract_obj_bundle` aceita os novos formatos
- [ ] Sem cor no viewer (princípio de fonte única respeitado)
