document.addEventListener('DOMContentLoaded', () => {
    //  Получение ссылок на элементы DOM 
    // Авторизация
    const authStatusDiv = document.getElementById('auth-status');
    const authFormPhone = document.getElementById('auth-form-phone');
    const authFormCode = document.getElementById('auth-form-code');
    const authFormPassword = document.getElementById('auth-form-password');
    const logoutButton = document.getElementById('logout-button');
    const messagesDiv = document.getElementById('messages');

    const phoneForm = document.getElementById('phone-form');
    const codeForm = document.getElementById('code-form');
    const passwordForm = document.getElementById('password-form');

    const phoneNumberInput = document.getElementById('phone_number');
    const codeInput = document.getElementById('phone_code');
    const passwordInput = document.getElementById('password_input');

    // Парсинг и каналы
    const parserControls = document.getElementById('parser-controls');
    const parsingStatusDiv = document.getElementById('parsing-status');
    const startParsingButton = document.getElementById('start-parsing-button');
    const stopParsingButton = document.getElementById('stop-parsing-button');

    const addChannelSection = document.getElementById('add-channel-section');
    const addChannelForm = document.getElementById('add-channel-form');
    const channelLinkInput = document.getElementById('channel_link');

    const channelsListSection = document.getElementById('channels-list-section');
    const channelsTableBody = document.getElementById('channels-table-body');


    //  Вспомогательные функции 

    /**
     * Добавляет сообщение в блок сообщений.
     * @param {string} text - Текст сообщения.
     * @param {string} [type='info'] - Тип сообщения ('info', 'success', 'error').
     */
    function appendMessage(text, type = 'info') {
        const messageElement = document.createElement('p');
        messageElement.textContent = text;
        messageElement.classList.add('message', type); // Добавляем классы для стилизации
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight; // Прокрутка к последнему сообщению
    }

    /**
     * Обновляет видимость форм и элементов интерфейса в зависимости от статуса авторизации.
     * @param {string} status - Статус авторизации ('authenticated', 'unauthenticated', 'code_sent', 'password_needed').
     * @param {string} message - Сообщение от сервера.
     */
    function updateUIForAuthStatus(status, message) {
        authStatusDiv.textContent = `Статус: ${status}`;
        appendMessage(message);

        // Скрыть все формы и элементы управления по умолчанию
        authFormPhone.style.display = 'none';
        authFormCode.style.display = 'none';
        authFormPassword.style.display = 'none';
        logoutButton.style.display = 'none';
        parserControls.style.display = 'none';
        addChannelSection.style.display = 'none';
        channelsListSection.style.display = 'none';
        messagesDiv.style.display = 'none'; // Скрываем, пока не авторизованы или нет сообщений

        switch (status) {
            case 'authenticated':
                // Если авторизованы, показываем элементы управления
                logoutButton.style.display = 'block';
                parserControls.style.display = 'block';
                addChannelSection.style.display = 'block';
                channelsListSection.style.display = 'block';
                messagesDiv.style.display = 'block';
                fetchChannels(); // Загрузить список каналов
                fetchParsingStatus(); // Загрузить статус парсинга
                break;
            case 'unauthenticated':
                // Если не авторизованы, показываем форму ввода телефона
                authFormPhone.style.display = 'block';
                messagesDiv.style.display = 'block'; // Показываем блок сообщений
                break;
            case 'code_sent':
                // Если код отправлен, показываем форму ввода кода
                authFormCode.style.display = 'block';
                messagesDiv.style.display = 'block'; // Показываем блок сообщений
                break;
            case 'password_needed':
                // Если требуется пароль 2FA, показываем форму ввода пароля
                authFormPassword.style.display = 'block';
                messagesDiv.style.display = 'block'; // Показываем блок сообщений
                break;
            default:
                // Неизвестный статус, по умолчанию показываем форму телефона
                authFormPhone.style.display = 'block';
                messagesDiv.style.display = 'block'; // Показываем блок сообщений
                appendMessage(`Неизвестный статус авторизации: ${status}`, 'error');
                break;
        }
    }


    //  Функции для работы с API 

    // Получение и отображение статуса авторизации с сервера
    async function fetchAuthStatus() {
        try {
            const response = await fetch('/auth-status');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            updateUIForAuthStatus(data.status, data.message);

        } catch (error) {
            console.error('Error fetching auth status:', error);
            authStatusDiv.textContent = 'Ошибка при загрузке статуса авторизации.';
            appendMessage('Ошибка при загрузке статуса авторизации: ' + error.message, 'error');
            // В случае ошибки сбросить к состоянию unauthenticated для возможности повторной попытки
            updateUIForAuthStatus('unauthenticated', 'Пожалуйста, попробуйте авторизоваться снова.');
        }
    }

    // Обработчик отправки номера телефона
    async function handlePhoneFormSubmit(event) {
        event.preventDefault(); // Предотвратить стандартную отправку формы
        const phoneNumber = phoneNumberInput.value;
        appendMessage('Отправка номера телефона...');

        try {
            const response = await fetch('/send-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone_number: phoneNumber }),
            });
            const data = await response.json();
            appendMessage(data.message, data.id === 'success' ? 'info' : 'error'); // Используем data.id для типа

            // После отправки номера и получения ответа, ОБЯЗАТЕЛЬНО ОБНОВИТЬ СТАТУС!
            fetchAuthStatus(); // Это ключевой момент для переключения формы

        } catch (error) {
            console.error('Error sending code:', error);
            appendMessage('Ошибка при отправке номера телефона: ' + error.message, 'error');
        }
    }

    // Обработчик подтверждения кода
    async function handleCodeFormSubmit(event) {
        event.preventDefault();
        const code = codeInput.value;
        appendMessage('Отправка кода...');

        try {
            const response = await fetch('/verify-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code: code }),
            });
            const data = await response.json();
            appendMessage(data.message, data.id === 'success' ? 'info' : 'error');
            
            fetchAuthStatus(); // Обновить статус после попытки верификации кода

        } catch (error) {
            console.error('Error verifying code:', error);
            appendMessage('Ошибка при подтверждении кода: ' + error.message, 'error');
        }
    }

    // Обработчик подтверждения пароля (двухфакторная аутентификация)
    async function handlePasswordFormSubmit(event) {
        event.preventDefault();
        const password = passwordInput.value;
        appendMessage('Отправка пароля...');

        try {
            const response = await fetch('/verify-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ password: password }),
            });
            const data = await response.json();
            appendMessage(data.message, data.id === 'success' ? 'info' : 'error');

            fetchAuthStatus(); // Обновить статус после попытки верификации пароля

        } catch (error) {
            console.error('Error verifying password:', error);
            appendMessage('Ошибка при подтверждении пароля: ' + error.message, 'error');
        }
    }

    // Обработчик выхода из аккаунта
    async function handleLogout() {
        appendMessage('Выход из Telegram...');
        try {
            const response = await fetch('/logout', {
                method: 'POST',
            });
            const data = await response.json();
            appendMessage(data.message, data.id === 'success' ? 'info' : 'error');
            fetchAuthStatus(); // Обновить статус после выхода
        } catch (error) {
            console.error('Error logging out:', error);
            appendMessage('Ошибка при выходе: ' + error.message, 'error');
        }
    }

    // Получение и отображение списка каналов
    async function fetchChannels() {
        try {
            const response = await fetch('/channels');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const channels = await response.json();
            
            channelsTableBody.innerHTML = ''; // Очистить таблицу перед добавлением
            if (channels.length === 0) {
                appendMessage('Каналы пока не добавлены.', 'info');
            } else {
                channels.forEach(channel => {
                    const row = channelsTableBody.insertRow();
                    row.insertCell(0).textContent = channel.link;
                    row.insertCell(1).textContent = channel.last_message_id || 'Неизвестно'; // Отображаем last_message_id

                    const deleteCell = row.insertCell(2);
                    const deleteButton = document.createElement('button');
                    deleteButton.textContent = 'Удалить';
                    deleteButton.classList.add('delete-button');
                    deleteButton.onclick = () => handleDeleteChannel(channel.id); // Передаем ID канала
                    deleteCell.appendChild(deleteButton);
                });
            }
        } catch (error) {
            console.error('Error fetching channels:', error);
            appendMessage('Ошибка при загрузке каналов: ' + error.message, 'error');
        }
    }

    // Обработчик добавления канала
    async function handleAddChannelSubmit(event) {
        event.preventDefault();
        const link = channelLinkInput.value.trim(); // Удаляем пробелы по краям
        if (!link) {
            appendMessage('Ссылка на канал не может быть пустой.', 'error');
            return;
        }
        appendMessage(`Добавление канала: ${link}...`);

        try {
            const response = await fetch('/add-channel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ link: link }),
            });
            const data = await response.json();
            appendMessage(data.message, data.id === 'success' ? 'info' : 'error');
            if (data.id === 'success') {
                channelLinkInput.value = ''; // Очистить поле ввода
                fetchChannels(); // Обновить список каналов
            }
        } catch (error) {
            console.error('Error adding channel:', error);
            appendMessage('Ошибка при добавлении канала: ' + error.message, 'error');
        }
    }

    // Обработчик удаления канала
    async function handleDeleteChannel(channelId) {
        if (!confirm('Вы уверены, что хотите удалить этот канал?')) {
            return;
        }
        appendMessage(`Удаление канала ID: ${channelId}...`);

        try {
            const response = await fetch('/delete-channel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ channel_id: channelId }),
            });
            const data = await response.json();
            appendMessage(data.message, data.id === 'success' ? 'info' : 'error');
            if (data.id === 'success') {
                fetchChannels(); // Обновить список каналов
            }
        } catch (error) {
            console.error('Error deleting channel:', error);
            appendMessage('Ошибка при удалении канала: ' + error.message, 'error');
        }
    }

    // Получение и отображение статуса парсинга
    async function fetchParsingStatus() {
        try {
            const response = await fetch('/parsing-status');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            if (data.is_parsing_active) {
                parsingStatusDiv.textContent = 'Статус парсинга: Активен';
                startParsingButton.disabled = true;
                stopParsingButton.disabled = false;
            } else {
                parsingStatusDiv.textContent = 'Статус парсинга: Неактивен';
                startParsingButton.disabled = false;
                stopParsingButton.disabled = true;
            }
        } catch (error) {
            console.error('Error fetching parsing status:', error);
            appendMessage('Ошибка при загрузке статуса парсинга: ' + error.message, 'error');
        }
    }

    // Обработчик запуска парсинга
    async function handleStartParsing() {
        appendMessage('Запуск парсинга...');
        try {
            const response = await fetch('/start-parsing', { method: 'POST' });
            const data = await response.json();
            appendMessage(data.message, data.id === 'success' ? 'info' : 'error');
            fetchParsingStatus(); // Обновить статус парсинга
        } catch (error) {
            console.error('Error starting parsing:', error);
            appendMessage('Ошибка при запуске парсинга: ' + error.message, 'error');
        }
    }

    // Обработчик остановки парсинга
    async function handleStopParsing() {
        appendMessage('Остановка парсинга...');
        try {
            const response = await fetch('/stop-parsing', { method: 'POST' });
            const data = await response.json();
            appendMessage(data.message, data.id === 'success' ? 'info' : 'error');
            fetchParsingStatus(); // Обновить статус парсинга
        } catch (error) {
            console.error('Error stopping parsing:', error);
            appendMessage('Ошибка при остановке парсинга: ' + error.message, 'error');
        }
    }

    //  Привязка обработчиков событий к элементам формы и кнопкам 
    // Используем console.error для отладки, если элемент не найден.

   // Если элемент формы телефона существует, добавляем слушатель события отправки
if (phoneForm) {
    phoneForm.addEventListener('submit', handlePhoneFormSubmit);
} else {
    // В противном случае выводим ошибку в консоль
    console.error('Элемент с ID "phone-form" не найден. Проверьте index.html');
}

// Если элемент формы кода существует, добавляем слушатель события отправки
if (codeForm) {
    codeForm.addEventListener('submit', handleCodeFormSubmit);
} else {
    // В противном случае выводим ошибку в консоль
    console.error('Элемент с ID "code-form" не найден. Проверьте index.html');
}

// Если элемент формы пароля существует, добавляем слушатель события отправки
if (passwordForm) {
    passwordForm.addEventListener('submit', handlePasswordFormSubmit);
} else {
    // В противном случае выводим ошибку в консоль
    console.error('Элемент с ID "password-form" не найден. Проверьте index.html');
}

// Если элемент кнопки выхода существует, добавляем слушатель события клика
if (logoutButton) {
    logoutButton.addEventListener('click', handleLogout);
} else {
    // В противном случае выводим ошибку в консоль
    console.error('Элемент с ID "logout-button" не найден. Проверьте index.html');
}

// Если элемент формы добавления канала существует, добавляем слушатель события отправки
if (addChannelForm) {
    addChannelForm.addEventListener('submit', handleAddChannelSubmit);
} else {
    // В противном случае выводим ошибку в консоль
    console.error('Элемент с ID "add-channel-form" не найден. Проверьте index.html');
}

// Если элемент кнопки начала парсинга существует, добавляем слушатель события клика
if (startParsingButton) {
    startParsingButton.addEventListener('click', handleStartParsing);
} else {
    // В противном случае выводим ошибку в консоль
    console.error('Элемент с ID "start-parsing-button" не найден. Проверьте index.html');
}

// Если элемент кнопки остановки парсинга существует, добавляем слушатель события клика
if (stopParsingButton) {
    stopParsingButton.addEventListener('click', handleStopParsing);
} else {
    // В противном случае выводим ошибку в консоль
    console.error('Элемент с ID "stop-parsing-button" не найден. Проверьте index.html');
}

    //  Начальная инициализация при загрузке страницы 
    fetchAuthStatus(); // Вызывается один раз при загрузке страницы для определения текущего статуса
});