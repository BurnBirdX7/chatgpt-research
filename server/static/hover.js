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
function getKey(elem) {
    const target = elem.closest(".category-table");
    return target.getAttribute('data-key');
}

async function updatePopup(event, target) {
    const source_url = target.getAttribute('data-source-url');
    const target_pos = target.getAttribute('data-target-pos');
    const target_likelihood = target.getAttribute('data-target-likelihood');
    const key = getKey(target);
    let color_class = 'color0';
    target.classList.forEach((cls) => {
        if (cls.startsWith('color')) {
            color_class = cls;
        }
    })

    // INNER HTML SECTION

    if (source_url != null) {
        popup.innerHTML = `
        <span class="${color_class}">⏺</span>
        <a href="${source_url}" target="_blank" class="source_reference">${source_url}</a><br>
        
        This token: \`<code>${target.innerText}</code>\`
        Matched token: \`<code>${target.getAttribute("data-source-token")}</code>\`
        
        <table>
        <tr><td><b>Target text</b></td><td>
            <span class="greytext">...${target.getAttribute('data-target-text-pre')}</span>${target.getAttribute('data-target-text')}<span class="greytext">${target.getAttribute('data-target-text-post')}...</span>
        </td></tr>
        <tr><td><b>Matched text</b></td><td>
            <span class="greytext">...${target.getAttribute('data-source-text-pre')}</span>${target.getAttribute('data-source-text')}<span class="greytext">${target.getAttribute('data-source-text-post')}...</span>
        </td></tr>    
        </table>
        <b>Score: </b> ${target.getAttribute('data-score')}
        <b>Pos: </b> ${target_pos}
        <b>Likelihood: </b> ${parseFloat(target_likelihood).toFixed(6)}
        <br>`
    } else {
        popup.innerHTML = `This token: \`<code>${target.innerText}</code>\`, Pos: ${target_pos} <br>`
    }

    if (loadImages) {
        popup.innerHTML += `
        <img alt="plot"
             class="likelihood_plot"
             src="/api/plots/${key}/${target_pos}?likelihood=${target_likelihood}">
        <br>`
    }

    if (displayTop10Chains) {
        if (target.top10 === undefined && !target.sent) {
            target.sent = true
            fetch(`/api/chains/${key}/${target_pos}`)
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

    if (displayDebugChains) {
        popup.innerHTML += `<pre>${target.getAttribute('data-chain').replaceAll('\t', '  ')}</pre>`
    }

    popup.innerHTML += "<button id=\"popup_close\">Unpin</button>"
    // ! INNER HTML SECTION

    popup.style.display = 'block';
    popup.style.left = (event.pageX + 10) + 'px';
    popup.style.top = (event.pageY - 10 - popup.offsetHeight) + 'px';
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

