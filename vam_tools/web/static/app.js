// VAM Tools - Complete MVP Application
const { createApp } = Vue;

createApp({
    data() {
        return {
            // Navigation
            currentView: 'dashboard',

            // Catalog management
            catalogs: [],
            currentCatalog: null,
            showCatalogManager: false,
            showAddCatalogForm: false,

            addCatalogForm: {
                name: '',
                source_directories: ''
            },

            // Jobs data
            allJobs: [],
            jobDetailsCache: {},
            jobsRefreshInterval: null,
            jobWebSockets: {}, // WebSocket connections by job ID

            // Worker health
            workerHealth: {
                status: 'unknown',
                workers: 0,
                active_tasks: 0,
                message: 'Checking...'
            },

            // Quick action forms
            showAnalyzeForm: false,

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

            // Browse (placeholder)
            images: [],
            currentFilter: 'all',
            searchQuery: '',

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
                return {
                    id: jobId,
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
        }
    },

    methods: {
        // Navigation
        setView(view) {
            this.currentView = view;
            if (view === 'jobs') {
                this.startJobsRefresh();
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
        async loadJobDetails(jobId) {
            try {
                const response = await axios.get(`/api/jobs/${jobId}`);
                this.jobDetailsCache[jobId] = response.data;

                // Connect WebSocket for active jobs
                const jobData = response.data;
                if ((jobData.status === 'PENDING' || jobData.status === 'PROGRESS') && !this.jobWebSockets[jobId]) {
                    this.connectJobWebSocket(jobId);
                }
            } catch (error) {
                console.error(`Error loading job ${jobId}:`, error);
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
                    this.loadJobDetails(response.data.job_id);
                }

                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit analysis job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
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

        loadImages() {
            this.addNotification('Image browsing not yet implemented', 'info');
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

            // Connect to Server-Sent Events endpoint
            const eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

            eventSource.onopen = () => {
                this.streamConnectionStatus = 'connected';
                this.streamConnectionStatusText = '● Connected';
            };

            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    // Update job status in streaming modal
                    if (data.status) {
                        this.streamingJob.status = data.status;
                    }
                    if (data.progress !== undefined) {
                        this.streamingJob.progress = data.progress.percent || 0;
                    }

                    // Add event to stream
                    const message = data.progress?.message || data.status || 'Update';
                    this.streamEvents.push({
                        timestamp: new Date(),
                        message: message,
                        status: data.status
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
                            eventSource.close();
                        }, 1000);
                    }
                } catch (e) {
                    console.error('Error parsing stream event:', e);
                }
            };

            eventSource.onerror = () => {
                this.streamConnectionStatus = 'error';
                this.streamConnectionStatusText = '● Connection Error';
                eventSource.close();
            };

            this.streamEventSource = eventSource;
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
        // Load catalogs first
        this.loadCatalogs().then(() => {
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
