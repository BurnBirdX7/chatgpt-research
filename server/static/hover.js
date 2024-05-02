// Popup setup:
const popup = document.createElement('div')
popup.className = 'popup'
popup.style.display = 'none'
document.body.appendChild(popup)

// Elements:
const sourceTexts = document.querySelectorAll('.src_text');
const sourceTextBoxes = document.querySelectorAll(".src_text_box");
const loadImagesCheckbox = document.querySelector("#load-images");
const displayDebugChainsCheckbox = document.querySelector("#display-debug-chains");
const displayTop10ChainsCheckbox = document.querySelector("#display-top-chains")

// Parameters:
let popupFixated = false;
let loadImages = loadImagesCheckbox.checked;
let displayDebugChains = displayDebugChainsCheckbox.checked;
let displayTop10Chains = displayTop10ChainsCheckbox.checked;


// Functions:
function getKeyAndType(elem) {
    const target = elem.closest(".category-block");
    return {
        key: target.getAttribute('data-key'),
        type: target.getAttribute('data-type'),
    };
}

function renderTop10(target) {
    if (target.top10 === undefined && !target.sent) {
        target.sent = true
        fetch(`/api/target-chains/${key}/${target_pos}`)
            .then(res => {
                if (!res.ok) {
                    throw new Error("Net OK")
                }
                return res.json()
            })
            .then((data) => {
                target.top10 = data;
            })
            .catch(console.error)
    }

    if (target.top10 !== undefined) {
        popup.innerHTML += '<ol>'
        for (let elem of target.top10) {
            popup.innerHTML += `<li>${elem.text} <b>score:</b> ${elem.score} <b>len:</b> ${elem.len}</li>`
        }
        popup.innerHTML += '</ol>'
    }
}


function renderDebugChains(target) {
    return `<pre>${target.getAttribute('data-chain').replaceAll('\t', '  ')}</pre>`
}

function renderGraph(key, target_pos, target_likelihood) {
    return `<img alt="plot"
          class="likelihood_plot"
          src="/api/plots/${key}/${target_pos}?likelihood=${target_likelihood}"><br>`
}

function renderSourceTypePopup(target, source_url, target_pos, target_likelihood, score) {
    const source_pos = target.getAttribute('data-source-pos');
    let color_class = 'color0';
    target.classList.forEach((cls) => {
        if (cls.startsWith('color')) {
            color_class = cls;
        }
    })

    return `<span class="${color_class}">⏺</span>
    <a href="${source_url}" target="_blank" class="source_reference">${source_url}</a><br>
    
    This token: \`<code>${target.innerText}</code>\` [${target_pos}]
    Matched token: \`<code>${target.getAttribute("data-source-token")}</code>\` [${source_pos}]
    
    <table>
    <tr><td><b>Target text</b></td><td>
        <span class="greytext">...${target.getAttribute('data-target-text-pre')}</span>${target.getAttribute('data-target-text')}<span class="greytext">${target.getAttribute('data-target-text-post')}...</span>
    </td></tr>
    <tr><td><b>Matched text</b></td><td>
        <span class="greytext">...${target.getAttribute('data-source-text-pre')}</span>${target.getAttribute('data-source-text')}<span class="greytext">${target.getAttribute('data-source-text-post')}...</span>
    </td></tr>    
    </table>
    <b>Score: </b> ${score}
    <b>Likelihood: </b> ${parseFloat(target_likelihood).toFixed(6)}
    <br>`
}

async function updatePopup(event, target) {
    const source_url = target.getAttribute('data-source-url');
    const target_pos = target.getAttribute('data-target-pos');
    const target_likelihood = target.getAttribute('data-target-likelihood');
    const score = target.getAttribute('data-score')
    const {key, type} = getKeyAndType(target);

    popup.innerHTML = "<i style='color: gray'>Click to pin</i><br>"

    if (source_url != null) {
        popup.innerHTML += renderSourceTypePopup(target, source_url, target_pos, target_likelihood, score)
    } else if (type === "source") {
        popup.innerHTML += `This token: \`<code>${target.innerText}</code>\` [${target_pos}] <br>`
    } else if (type === "score") {
        popup.innerHTML += `This token: \`<code>${target.innerText}</code>\`, pos: ${target_pos}, score: ${score} <br>`
    } else {
        popup.innerHTML += "Unsupported token type"
    }

    if (loadImages) {
        popup.innerHTML += renderGraph(key, target_pos, target_likelihood)
    }

    if (type === "source" && displayTop10Chains) {
        renderTop10(target)
    }

    if (type === "source" && displayDebugChains) {
        popup.innerHTML += renderDebugChains(target)
    }

    popup.innerHTML += "<button id=\"popup_close\">Unpin</button>"
    // ! INNER HTML SECTION

    popup.style.display = 'block';
    popup.style.left = Math.min(event.pageX + 10, window.innerWidth - popup.offsetWidth) + 'px';
    popup.style.top = Math.max(0, event.pageY - 10 - popup.offsetHeight) + 'px';
    popup.querySelector("#popup_close").addEventListener("click", () => {
        popupFixated = false;
        popup.style.display = 'none';
    })
}

// Add Listeners

loadImagesCheckbox.addEventListener('change', () => {
    loadImages = loadImagesCheckbox.checked;
})

displayDebugChainsCheckbox.addEventListener('change', () => {
    displayDebugChains = displayDebugChainsCheckbox.checked;
})

displayTop10ChainsCheckbox.addEventListener('change', () => {
    displayTop10Chains = displayTop10ChainsCheckbox.checked;
})

sourceTexts.forEach(span => {
    span.addEventListener('mousemove', (event) => {
        if (popupFixated) {
            return
        }

        updatePopup(event, event.target)
    })

    span.addEventListener('click', (event) => {
        updatePopup(event, event.target)
        popupFixated = true
    })
})

sourceTextBoxes.forEach(box => {
    box.addEventListener('mouseout', event => {
        if (!popupFixated) {
            popup.style.display = 'none'
        }
    })
})

