document.addEventListener("DOMContentLoaded", () => {
    const enterButton = document.getElementById("enterPlatform");
    const introScreen = document.getElementById("introScreen");
    const reportButton = document.getElementById("downloadReport");
    const reportText = document.getElementById("reportText");

    if (enterButton && introScreen) {
        enterButton.addEventListener("click", () => {
            document.body.classList.add("app-open");
            window.setTimeout(() => {
                introScreen.setAttribute("aria-hidden", "true");
            }, 850);
        });
    }

    if (reportButton && reportText) {
        reportButton.addEventListener("click", () => {
            const blob = new Blob([reportText.value], { type: "text/plain;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "saudi-labor-analysis-report.txt";
            link.click();
            URL.revokeObjectURL(url);
        });
    }

    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach((input) => {
        input.addEventListener("change", () => {
            const helper = input.parentElement?.querySelector("small");
            const fileName = input.files?.[0]?.name;
            if (helper && fileName) {
                helper.textContent = `تم اختيار الملف: ${fileName}`;
            }
        });
    });
});
