// VAM Tools - Complete MVP Application
const { createApp } = Vue;

createApp({
    data() {
        return {
            // Navigation
            currentView: 'browse',

            // Catalog management
            catalogs: [],
            currentCatalog: null,
            showCatalogManager: false,
            showAddCatalogForm: false,
            showEditCatalogForm: false,

            addCatalogForm: {
                name: '',
                source_directories: ''
            },

            editCatalogForm: {
                id: null,
                name: '',
                source_directories: ''
            },

            // Jobs data
            allJobs: [],
            jobDetailsCache: {},
            jobMetadata: {}, // Store job name and created timestamp
            jobProgressTracking: {}, // Track last progress update time and value
            jobsRefreshInterval: null,
            jobWebSockets: {}, // WebSocket connections by job ID
            STUCK_JOB_TIMEOUT: 5 * 60 * 1000, // 5 minutes in milliseconds

            // Worker health
            workerHealth: {
                status: 'unknown',
                workers: 0,
                active_tasks: 0,
                message: 'Checking...'
            },

            // Quick action forms
            showAnalyzeForm: false,
            showPathHelp: false,

            analyzeForm: {
                catalog_id: null,
                detect_duplicates: false,
                force_reanalyze: false
            },

            // Notifications
            notifications: [],

            // Review Queue (placeholder)
            reviewQueue: [],
            reviewQueueStats: {
                total_needing_review: 0,
                date_conflicts: 0,
                no_date: 0,
                suspicious_dates: 0,
            },

            // Browse
            images: [],
            imagesTotal: 0,
            imagesLoading: false,
            imagesPage: 0,
            imagesPageSize: 50,
            currentFilter: 'all',
            searchQuery: '',
            thumbnailSize: 'medium',
            selectedImages: new Set(),

            // Lightbox
            lightboxVisible: false,
            lightboxImageIndex: 0,
            lightboxImage: null,
            lightboxZoom: 1,

            // Job streaming
            streamingJob: null,
            streamEvents: [],
            streamEventSource: null,
            streamConnectionStatus: null,
            streamConnectionStatusText: '',
        };
    },

    computed: {
        filteredImages() {
            return this.images;
        },

        jobs() {
            return this.allJobs.map(jobId => {
                const details = this.jobDetailsCache[jobId] || {};
                const metadata = this.jobMetadata[jobId];
                return {
                    id: jobId,
                    name: metadata?.name || 'Job',
                    started: metadata?.started || null,
                    status: details.status || 'PENDING',
                    progress: details.progress || {},
                    result: details.result || {},
                };
            });
        },

        activeJobs() {
            return this.jobs.filter(job =>
                job.status === 'PENDING' || job.status === 'PROGRESS'
            );
        },

        hasActiveJobs() {
            return this.activeJobs.length > 0;
        },

        hasMoreImages() {
            return this.images.length < this.imagesTotal;
        },

        thumbnailGridClass() {
            const sizeMap = {
                'small': 'grid-6',
                'medium': 'grid-4',
                'large': 'grid-3'
            };
            return sizeMap[this.thumbnailSize] || 'grid-4';
        }
    },

    methods: {
        // Navigation
        setView(view) {
            this.currentView = view;
            if (view === 'jobs') {
                this.startJobsRefresh();
            } else if (view === 'browse') {
                if (this.images.length === 0) {
                    this.loadImages(true);
                }
            }
        },

        // Catalog Management
        async loadCatalogs() {
            try {
                const response = await axios.get('/api/catalogs/');
                this.catalogs = response.data;

                // Set first catalog as current if available
                if (this.catalogs.length > 0 && !this.currentCatalog) {
                    this.currentCatalog = this.catalogs[0];
                }

                // Set form defaults
                if (this.currentCatalog) {
                    this.analyzeForm.catalog_id = this.currentCatalog.id;
                }
            } catch (error) {
                console.error('Error loading catalogs:', error);
                this.addNotification('Failed to load catalogs', 'error');
            }
        },

        switchCatalog(catalog) {
            this.currentCatalog = catalog;
            this.analyzeForm.catalog_id = catalog.id;
            this.addNotification(`Switched to catalog: ${catalog.name}`, 'success');

            // Load images for the selected catalog if in browse view
            if (this.currentView === 'browse') {
                this.loadImages(true);
            }
        },

        async submitAddCatalog() {
            try {
                const sourceDirs = this.addCatalogForm.source_directories
                    .split('\n')
                    .map(s => s.trim())
                    .filter(s => s.length > 0);

                await axios.post('/api/catalogs/', {
                    name: this.addCatalogForm.name,
                    source_directories: sourceDirs
                });

                this.addNotification('Catalog added successfully', 'success');
                this.showAddCatalogForm = false;

                // Reset form
                this.addCatalogForm = {
                    name: '',
                    source_directories: ''
                };

                await this.loadCatalogs();
            } catch (error) {
                this.addNotification('Failed to add catalog: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },

        openEditCatalog(catalog) {
            this.editCatalogForm = {
                id: catalog.id,
                name: catalog.name,
                source_directories: catalog.source_directories.join('\n')
            };
            this.showEditCatalogForm = true;
            this.showCatalogManager = false;
        },

        async submitEditCatalog() {
            try {
                const sourceDirs = this.editCatalogForm.source_directories
                    .split('\n')
                    .map(s => s.trim())
                    .filter(s => s.length > 0);

                await axios.put(`/api/catalogs/${this.editCatalogForm.id}`, {
                    name: this.editCatalogForm.name,
                    source_directories: sourceDirs
                });

                this.addNotification('Catalog updated successfully', 'success');
                this.showEditCatalogForm = false;

                // Reset form
                this.editCatalogForm = {
                    id: null,
                    name: '',
                    source_directories: ''
                };

                await this.loadCatalogs();
            } catch (error) {
                this.addNotification('Failed to update catalog: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },

        async deleteCatalog(catalogId) {
            if (!confirm('Are you sure you want to delete this catalog? This will delete all catalog data.')) {
                return;
            }

            try {
                await axios.delete(`/api/catalogs/${catalogId}`);
                this.addNotification('Catalog deleted', 'info');
                if (this.currentCatalog && this.currentCatalog.id === catalogId) {
                    this.currentCatalog = null;
                }
                await this.loadCatalogs();
            } catch (error) {
                this.addNotification('Failed to delete catalog', 'error');
                console.error(error);
            }
        },

        // Job methods
        async loadPersistedJobs() {
            try {
                // Load jobs from API (database)
                const response = await axios.get('/api/jobs/');
                const jobs = response.data;

                // Extract job IDs and set metadata
                this.allJobs = jobs.map(job => job.id);

                // Build job metadata from DB data
                jobs.forEach(job => {
                    this.jobMetadata[job.id] = {
                        name: job.job_type,
                        created: job.created_at
                    };

                    // Cache the full job details
                    this.jobDetailsCache[job.id] = {
                        job_id: job.id,
                        status: job.status,
                        progress: {},
                        result: job.result || {}
                    };
                });

                // Load latest details for all jobs from Celery
                this.allJobs.forEach(jobId => this.loadJobDetails(jobId));
            } catch (error) {
                console.error('Error loading persisted jobs:', error);
            }
        },

        persistJobs() {
            // No-op: Jobs are now persisted to database automatically via API
        },

        async loadJobDetails(jobId) {
            try {
                const response = await axios.get(`/api/jobs/${jobId}`);
                const jobData = response.data;
                this.jobDetailsCache[jobId] = jobData;

                // Track progress for stuck job detection
                if (jobData.status === 'PROGRESS' && jobData.progress) {
                    const progressKey = JSON.stringify(jobData.progress);
                    const now = Date.now();

                    if (!this.jobProgressTracking[jobId]) {
                        // First time seeing this job
                        this.jobProgressTracking[jobId] = {
                            lastProgressKey: progressKey,
                            lastUpdateTime: now
                        };
                    } else {
                        const tracking = this.jobProgressTracking[jobId];

                        // Check if progress has changed
                        if (tracking.lastProgressKey !== progressKey) {
                            // Progress changed, update tracking
                            tracking.lastProgressKey = progressKey;
                            tracking.lastUpdateTime = now;
                        } else {
                            // Progress hasn't changed, check if stuck
                            const timeSinceUpdate = now - tracking.lastUpdateTime;
                            if (timeSinceUpdate > this.STUCK_JOB_TIMEOUT) {
                                console.warn(`Job ${jobId} appears stuck (no progress for ${Math.round(timeSinceUpdate/1000/60)} minutes), auto-revoking`);
                                this.addNotification(`Job ${this.jobMetadata[jobId]?.name || jobId} stuck - auto-canceling`, 'warning');
                                await this.revokeJob(jobId, true); // Force terminate
                                return;
                            }
                        }
                    }
                }

                // Connect WebSocket for active jobs
                if ((jobData.status === 'PENDING' || jobData.status === 'PROGRESS') && !this.jobWebSockets[jobId]) {
                    this.connectJobWebSocket(jobId);
                }
            } catch (error) {
                // If job not found (404), remove it from tracked jobs
                if (error.response && error.response.status === 404) {
                    console.log(`Job ${jobId} no longer exists, removing from tracked jobs`);
                    this.removeJob(jobId);
                } else {
                    console.error(`Error loading job ${jobId}:`, error);
                }
            }
        },

        connectJobWebSocket(jobId) {
            // Don't reconnect if already connected
            if (this.jobWebSockets[jobId]) return;

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/jobs/${jobId}/stream`;

            console.log(`Connecting WebSocket for job ${jobId}...`);
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log(`WebSocket connected for job ${jobId}`);
            };

            ws.onmessage = (event) => {
                const update = JSON.parse(event.data);
                console.log(`Job ${jobId} update:`, update);

                // Update job details cache
                this.jobDetailsCache[jobId] = {
                    job_id: update.job_id,
                    status: update.status,
                    progress: update.progress || {},
                    result: update.result || {}
                };

                // Close WebSocket if job is done
                if (update.status === 'SUCCESS' || update.status === 'FAILURE') {
                    this.disconnectJobWebSocket(jobId);

                    // Show notification
                    if (update.status === 'SUCCESS') {
                        this.addNotification(`Job ${jobId.substring(0, 8)} completed successfully`, 'success');
                    } else {
                        this.addNotification(`Job ${jobId.substring(0, 8)} failed`, 'error');
                    }
                }
            };

            ws.onerror = (error) => {
                console.error(`WebSocket error for job ${jobId}:`, error);
                this.disconnectJobWebSocket(jobId);
            };

            ws.onclose = () => {
                console.log(`WebSocket closed for job ${jobId}`);
                delete this.jobWebSockets[jobId];
            };

            this.jobWebSockets[jobId] = ws;
        },

        disconnectJobWebSocket(jobId) {
            const ws = this.jobWebSockets[jobId];
            if (ws) {
                ws.close();
                delete this.jobWebSockets[jobId];
            }
        },

        removeJob(jobId) {
            // Remove from tracked jobs list
            const index = this.allJobs.indexOf(jobId);
            if (index > -1) {
                this.allJobs.splice(index, 1);
            }

            // Clean up job details cache
            delete this.jobDetailsCache[jobId];

            // Clean up job metadata
            delete this.jobMetadata[jobId];

            // Clean up progress tracking
            delete this.jobProgressTracking[jobId];

            // Disconnect websocket if open
            this.disconnectJobWebSocket(jobId);

            // Persist the updated job list
            this.persistJobs();

            console.log(`Removed job ${jobId} from tracked jobs`);
        },

        async revokeJob(jobId, terminate = false) {
            try {
                await axios.delete(`/api/jobs/${jobId}`, {
                    params: { terminate }
                });

                // Remove from tracked jobs
                this.removeJob(jobId);

                const jobName = this.jobMetadata[jobId]?.name || jobId.substring(0, 8);
                this.addNotification(`Job ${jobName} canceled`, 'info');
            } catch (error) {
                console.error(`Failed to revoke job ${jobId}:`, error);
                this.addNotification('Failed to cancel job', 'error');
            }
        },

        async loadWorkerHealth() {
            try {
                const response = await axios.get('/api/jobs/health');
                this.workerHealth = response.data;
            } catch (error) {
                this.workerHealth = {
                    status: 'error',
                    workers: 0,
                    active_tasks: 0,
                    message: 'Failed to check worker health'
                };
            }
        },

        async submitAnalyzeJob() {
            try {
                const catalog = this.catalogs.find(c => c.id === this.analyzeForm.catalog_id);
                if (!catalog) {
                    this.addNotification('Please select a catalog', 'error');
                    return;
                }

                const response = await axios.post('/api/jobs/analyze', {
                    catalog_id: catalog.id,
                    source_directories: catalog.source_directories,
                    detect_duplicates: this.analyzeForm.detect_duplicates,
                    force_reanalyze: this.analyzeForm.force_reanalyze
                });

                this.addNotification('Analysis job submitted successfully', 'success');
                this.showAnalyzeForm = false;

                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    // Store job metadata
                    this.jobMetadata[response.data.job_id] = {
                        name: `Analyze: ${catalog.name}`,
                        started: new Date().toISOString()
                    };
                    this.persistJobs(); // Save to localStorage
                    this.loadJobDetails(response.data.job_id);
                }

                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit analysis job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },

        async startScan() {
            try {
                if (!this.currentCatalog) {
                    this.addNotification('Please select a catalog', 'error');
                    return;
                }

                const response = await axios.post('/api/jobs/scan', {
                    catalog_id: this.currentCatalog.id,
                    directories: this.currentCatalog.source_directories
                });

                this.addNotification('Scan job submitted successfully', 'success');

                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    // Store job metadata
                    this.jobMetadata[response.data.job_id] = {
                        name: `Scan: ${this.currentCatalog.name}`,
                        started: new Date().toISOString()
                    };
                    this.persistJobs(); // Save to localStorage
                    this.loadJobDetails(response.data.job_id);
                }

                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit scan job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },

        async startFindDuplicates() {
            try {
                if (!this.currentCatalog) {
                    this.addNotification('Please select a catalog', 'error');
                    return;
                }

                const response = await axios.post('/api/jobs/analyze', {
                    catalog_id: this.currentCatalog.id,
                    source_directories: this.currentCatalog.source_directories,
                    detect_duplicates: true,
                    force_reanalyze: false
                });

                this.addNotification('Duplicate detection job submitted successfully', 'success');

                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    // Store job metadata
                    this.jobMetadata[response.data.job_id] = {
                        name: `Find Duplicates: ${this.currentCatalog.name}`,
                        started: new Date().toISOString()
                    };
                    this.persistJobs(); // Save to localStorage
                    this.loadJobDetails(response.data.job_id);
                }

                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit duplicate detection job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },

        startFindFaces() {
            this.addNotification('Coming soon', 'info');
        },

        startJobsRefresh() {
            if (this.jobsRefreshInterval) return;

            this.loadWorkerHealth();
            this.jobsRefreshInterval = setInterval(() => {
                // Reload job details for tracked jobs
                this.allJobs.forEach(jobId => {
                    this.loadJobDetails(jobId);
                });
                this.loadWorkerHealth();
            }, 2000);
        },

        stopJobsRefresh() {
            if (this.jobsRefreshInterval) {
                clearInterval(this.jobsRefreshInterval);
                this.jobsRefreshInterval = null;
            }
        },

        getJobStatusClass(status) {
            const statusClasses = {
                'PENDING': 'status-pending',
                'PROGRESS': 'status-progress',
                'SUCCESS': 'status-success',
                'FAILURE': 'status-failure'
            };
            return statusClasses[status] || 'status-unknown';
        },

        formatJobStatus(status) {
            const statusLabels = {
                'PENDING': 'Queued',
                'PROGRESS': 'Running',
                'SUCCESS': 'Completed',
                'FAILURE': 'Failed'
            };
            return statusLabels[status] || status;
        },

        // Stub methods for features not yet implemented
        cancelJob(jobId) {
            this.addNotification('Job cancel not yet implemented', 'info');
        },

        killJob(jobId) {
            this.addNotification('Job kill not yet implemented', 'info');
        },

        rerunJob(jobId) {
            this.addNotification('Job rerun not yet implemented', 'info');
        },

        async loadImages(reset = false) {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog first', 'info');
                return;
            }

            if (reset) {
                this.images = [];
                this.imagesPage = 0;
                this.imagesTotal = 0;
            }

            this.imagesLoading = true;

            try {
                const offset = this.imagesPage * this.imagesPageSize;
                const response = await axios.get(`/api/catalogs/${this.currentCatalog.id}/images`, {
                    params: {
                        limit: this.imagesPageSize,
                        offset: offset
                    }
                });

                if (reset) {
                    this.images = response.data.images;
                } else {
                    this.images.push(...response.data.images);
                }

                this.imagesTotal = response.data.total;
                this.imagesPage++;
            } catch (error) {
                console.error('Error loading images:', error);
                this.addNotification('Failed to load images', 'error');
            } finally {
                this.imagesLoading = false;
            }
        },

        getThumbnailUrl(image) {
            if (!this.currentCatalog) return '';
            return `/api/catalogs/${this.currentCatalog.id}/images/${image.id}/thumbnail?size=${this.thumbnailSize}`;
        },

        toggleImageSelection(imageId) {
            if (this.selectedImages.has(imageId)) {
                this.selectedImages.delete(imageId);
            } else {
                this.selectedImages.add(imageId);
            }
        },

        isImageSelected(imageId) {
            return this.selectedImages.has(imageId);
        },

        clearSelection() {
            this.selectedImages.clear();
        },

        // Lightbox methods
        openLightbox(imageIndex) {
            this.lightboxImageIndex = imageIndex;
            this.lightboxImage = this.images[imageIndex];
            this.lightboxVisible = true;
            this.lightboxZoom = 1;

            // Add keyboard event listener with proper binding
            this._boundKeydownHandler = this.handleLightboxKeydown.bind(this);
            document.addEventListener('keydown', this._boundKeydownHandler);
        },

        closeLightbox() {
            this.lightboxVisible = false;
            this.lightboxImage = null;
            this.lightboxZoom = 1;

            // Remove keyboard event listener
            if (this._boundKeydownHandler) {
                document.removeEventListener('keydown', this._boundKeydownHandler);
                this._boundKeydownHandler = null;
            }
        },

        nextImage() {
            if (this.lightboxImageIndex < this.images.length - 1) {
                this.lightboxImageIndex++;
                this.lightboxImage = this.images[this.lightboxImageIndex];
                this.lightboxZoom = 1;
            }
        },

        prevImage() {
            if (this.lightboxImageIndex > 0) {
                this.lightboxImageIndex--;
                this.lightboxImage = this.images[this.lightboxImageIndex];
                this.lightboxZoom = 1;
            }
        },

        handleLightboxKeydown(event) {
            if (!this.lightboxVisible) return;

            switch (event.key) {
                case 'Escape':
                    this.closeLightbox();
                    break;
                case 'ArrowLeft':
                    this.prevImage();
                    break;
                case 'ArrowRight':
                    this.nextImage();
                    break;
            }
        },

        getFullImageUrl(image) {
            if (!this.currentCatalog || !image) return '';
            return `/api/catalogs/${this.currentCatalog.id}/images/${image.id}/thumbnail?size=large&quality=95`;
        },

        formatFileSize(bytes) {
            if (!bytes) return 'Unknown';
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(1024));
            return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
        },

        formatDate(dateStr) {
            if (!dateStr) return 'Unknown';
            try {
                return new Date(dateStr).toLocaleString();
            } catch (e) {
                return dateStr;
            }
        },

        resolveReviewItem(itemId) {
            this.addNotification('Review queue not yet implemented', 'info');
        },

        skipReviewItem(itemId) {
            this.addNotification('Review queue not yet implemented', 'info');
        },

        // Job Streaming
        showJobStream(jobId) {
            const job = this.jobs.find(j => j.id === jobId);
            if (!job) return;

            this.streamingJob = job;
            this.streamEvents = [];
            this.streamConnectionStatus = 'connecting';
            this.streamConnectionStatusText = 'Connecting...';

            // Connect to WebSocket endpoint
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/jobs/${jobId}/stream`;

            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                this.streamConnectionStatus = 'connected';
                this.streamConnectionStatusText = '● Connected';
                console.log(`WebSocket connected for streaming job ${jobId}`);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    // Update job status in streaming modal
                    if (data.status) {
                        this.streamingJob.status = data.status;
                    }
                    if (data.progress !== undefined) {
                        this.streamingJob.progress = data.progress;
                    }

                    // Add event to stream
                    const message = data.progress?.message || data.status || 'Update';
                    this.streamEvents.push({
                        timestamp: new Date(),
                        message: message,
                        status: data.status,
                        data: data
                    });

                    // Auto-scroll to bottom
                    this.$nextTick(() => {
                        const output = this.$refs.streamOutput;
                        if (output) {
                            output.scrollTop = output.scrollHeight;
                        }
                    });

                    // Close stream if job complete
                    if (data.status === 'SUCCESS' || data.status === 'FAILURE') {
                        this.streamConnectionStatus = 'completed';
                        this.streamConnectionStatusText = '● Completed';
                        setTimeout(() => {
                            ws.close();
                        }, 1000);
                    }
                } catch (e) {
                    console.error('Error parsing stream message:', e);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.streamConnectionStatus = 'error';
                this.streamConnectionStatusText = '● Connection Error';
            };

            ws.onclose = () => {
                if (this.streamConnectionStatus !== 'completed') {
                    this.streamConnectionStatus = 'disconnected';
                    this.streamConnectionStatusText = '● Disconnected';
                }
            };

            this.streamEventSource = ws;
        },

        closeJobStream() {
            if (this.streamEventSource) {
                this.streamEventSource.close();
                this.streamEventSource = null;
            }
            this.streamingJob = null;
            this.streamEvents = [];
            this.streamConnectionStatus = null;
        },

        clearStreamEvents() {
            this.streamEvents = [];
        },

        // Notifications
        addNotification(message, type = 'info') {
            const id = Date.now();
            this.notifications.push({ id, message, type });
            setTimeout(() => {
                this.notifications = this.notifications.filter(n => n.id !== id);
            }, 5000);
        },

        removeNotification(id) {
            this.notifications = this.notifications.filter(n => n.id !== id);
        },

        // Formatting
        formatTime(date) {
            if (!date) return '';
            return new Date(date).toLocaleTimeString();
        },

        // Formatting
        formatNumber(num) {
            return new Intl.NumberFormat().format(num || 0);
        },

        formatDate(dateStr) {
            if (!dateStr) return 'Unknown';
            return new Date(dateStr).toLocaleString();
        },

        displayPath(path) {
            // Display user-friendly paths
            if (!path) return '';
            if (path.startsWith('/app/')) {
                return '~/' + path.substring(5);
            }
            if (path.startsWith('/host/home/')) {
                return path.replace('/host/home/', '/home/');
            }
            if (path.startsWith('/host/synology/')) {
                return '/mnt/synology/' + path.substring(15);
            }
            return path;
        }
    },

    mounted() {
        // Load persisted jobs from localStorage
        this.loadPersistedJobs();

        // Load catalogs first
        this.loadCatalogs().then(() => {
            // Auto-select first catalog if available
            if (this.catalogs.length > 0) {
                this.switchCatalog(this.catalogs[0]);
            }

            // Start monitoring worker health
            this.startJobsRefresh();
        });
    },

    beforeUnmount() {
        this.stopJobsRefresh();

        // Close all WebSocket connections
        Object.keys(this.jobWebSockets).forEach(jobId => {
            this.disconnectJobWebSocket(jobId);
        });
    }
}).mount('#app');
