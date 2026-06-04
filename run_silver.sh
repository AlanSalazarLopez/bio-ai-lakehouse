#!/bin/bash
# run_silver.sh — wrapper con retry automático por perfil
#
# Lanza silver_transform.py empezando por el perfil más agresivo
# que detecte según la RAM. Si falla por OOM (exit 137 = SIGKILL del kernel),
# baja automáticamente al perfil inferior y reintenta.
#
# Exit codes del pipeline:
#   0   → éxito
#   1   → quality gate falló (PipelineQualityError) — no reintentar
#   2   → bug en el código (excepción no manejada) — no reintentar
#   137 → OOM: kernel mató el proceso — reintentar con perfil inferior
#
# Uso:
#   bash run_silver.sh                  # detección automática
#   bash run_silver.sh --profile BALANCED  # forzar perfil inicial

set -euo pipefail

CONTAINER="spark-master"
WORKDIR="/opt/spark/work-dir"
SCRIPT="src/jobs/silver_transform.py"
PROFILES=("PRO" "PERFORMANCE" "BALANCED" "SURVIVAL")
MAX_ATTEMPTS=${#PROFILES[@]}

# Si se pasa --profile, empezar desde ese perfil
FORCE_PROFILE=""
if [[ "${1:-}" == "--profile" && -n "${2:-}" ]]; then
    FORCE_PROFILE=$(echo "$2" | tr '[:lower:]' '[:upper:]')
    # Validar que el perfil existe
    VALID=false
    for p in "${PROFILES[@]}"; do
        [[ "$p" == "$FORCE_PROFILE" ]] && VALID=true && break
    done
    if [[ "$VALID" == false ]]; then
        echo "❌ Perfil '$FORCE_PROFILE' no válido. Opciones: ${PROFILES[*]}"
        exit 2
    fi
    echo "🔧 Perfil forzado: $FORCE_PROFILE"
fi

echo "════════════════════════════════════════════"
echo "  Bio-AI Lakehouse — Silver Transform"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════"

# Determinar índice de inicio
START_IDX=0
if [[ -n "$FORCE_PROFILE" ]]; then
    for i in "${!PROFILES[@]}"; do
        [[ "${PROFILES[$i]}" == "$FORCE_PROFILE" ]] && START_IDX=$i && break
    done
fi

ATTEMPT=0
for (( i=START_IDX; i<MAX_ATTEMPTS; i++ )); do
    PROFILE="${PROFILES[$i]}"
    ATTEMPT=$((ATTEMPT + 1))

    echo ""
    echo "▶ Intento $ATTEMPT — Perfil: $PROFILE"
    echo "  $(date '+%H:%M:%S')"
    echo "────────────────────────────────────────────"

    # Ejecutar en el contenedor
    docker exec \
        --workdir "$WORKDIR" \
        "$CONTAINER" \
        env PYTHONPATH=. \
        python3 "$SCRIPT" --profile "$PROFILE"

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "✅ Pipeline completado exitosamente con perfil: $PROFILE"
        echo "   $(date '+%H:%M:%S')"
        exit 0

    elif [[ $EXIT_CODE -eq 137 ]]; then
        echo ""
        echo "⚠️  OOM detectado (exit 137) con perfil: $PROFILE"

        if [[ $i -lt $((MAX_ATTEMPTS - 1)) ]]; then
            NEXT="${PROFILES[$((i+1))]}"
            echo "   Bajando a perfil: $NEXT"
            echo "   Reintentando en 5 segundos..."
            sleep 5
        else
            echo "❌ Ya estamos en SURVIVAL y sigue fallando por OOM."
            echo "   Revisa cuánta RAM tiene disponible el contenedor."
            exit 137
        fi

    elif [[ $EXIT_CODE -eq 1 ]]; then
        echo ""
        echo "❌ Quality gate falló (exit 1) — no se reintenta."
        echo "   Revisa los logs de quality_checks.py."
        exit 1

    else
        echo ""
        echo "❌ Error de código (exit $EXIT_CODE) — no se reintenta."
        echo "   Revisa el stack trace arriba."
        exit $EXIT_CODE
    fi
done