const popup = document.createElement('div')
popup.className = 'popup'
popup.style.display = 'none'
document.body.appendChild(popup)

const source_texts = document.querySelectorAll('.src_text')
const source_text_boxes = document.querySelectorAll(".src_text_box")
const timer = clearTimeout(12)

let popupFixated = false

function updatePopup(event, target) {
        const source_url = target.getAttribute('data-source')
        const chain_text = target.getAttribute('data-chain')
        const source_text = target.getAttribute('data-sourcetext')
        let color_class = 'color0'
        target.classList.forEach((cls) => {
            if (cls.startsWith('color')) {
                color_class = cls
            }
        })

        popup.innerHTML = `
        <span class="${color_class}">⏺</span>
        <a href="${source_url}" target="_blank" class="source_reference">${source_url}</a><br>
        <table>
        <tr><td><b>This text</b></td><td>${target.innerText}</td></tr>
        <tr><td><b>Matched text</b></td><td>${source_text}</td></tr>    
        </table>
        <pre>${chain_text}</pre><br>
        <button id="popup_close">Close</button>
        `
        popup.style.display = 'block'
        popup.style.left = (event.pageX + 10) + 'px'
        popup.style.top = (event.pageY + 10 - popup.offsetHeight) + 'px'
        popup.querySelector("#popup_close").addEventListener("click", () => {
            popupFixated = false
            popup.style.display = 'none'
        })
}

source_texts.forEach(span => {
    span.addEventListener('mouseenter', (event) => {
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

source_text_boxes.forEach(box => {
    box.addEventListener('mouseout', event => {
        if (!popupFixated) {
            popup.style.display = 'none'
        }
    })
})

