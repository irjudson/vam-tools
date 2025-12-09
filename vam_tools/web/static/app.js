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
            archivedJobs: new Set(), // Track collapsed/archived job IDs
            STUCK_JOB_TIMEOUT: 30 * 60 * 1000, // 30 minutes in milliseconds (long-running jobs like duplicate detection can take time)

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
            showScanConfirmModal: false,
            showAutoTagConfirmModal: false,
            showDuplicatesConfirmModal: false,

            // Pipeline options
            scanContinuePipeline: false,
            autoTagContinuePipeline: false,
            autoTagBackend: 'openclip',
            autoTagMode: 'untagged_only',

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

            // Catalog statistics
            catalogStats: null,
            catalogStatsLoading: false,
            statsExpanded: false,

            // Browse
            images: [],
            imagesTotal: 0,
            imagesLoading: false,
            imagesPage: 0,
            imagesPageSize: 100,
            currentFilter: 'all',
            searchQuery: '',
            thumbnailSize: 'medium',
            selectedImages: new Set(),
            infiniteScrollEnabled: true,
            scrollThreshold: 300, // pixels from bottom to trigger load

            // Search, Filter, Sort
            filterOptions: null,
            filtersExpanded: false,
            filters: {
                search: '',
                file_type: '',
                camera_make: '',
                camera_model: '',
                lens: '',
                focal_length: '',
                f_stop: '',
                has_gps: null,
                date_from: '',
                date_to: '',
                // Tag filters
                tags: '',
                tag_match: 'any',
                has_tags: null
            },
            // Available tags for filtering
            availableTags: [],
            tagsLoading: false,
            sortBy: 'date',
            sortOrder: 'desc',
            searchDebounceTimer: null,
            // Unified Search
            searchQuery: '',
            semanticSearchQuery: '', // Keep for backward compatibility
            searchResults: [],
            isSearching: false,
            searchMode: false,
            lastSearchType: null, // 'text' or 'semantic'

            // Lightbox
            lightboxVisible: false,
            lightboxImageIndex: 0,
            lightboxImage: null,
            lightboxZoom: 1,

            // 3-Column Layout Panels
            showLeftPanel: true,
            showRightPanel: true,
            selectedImage: null,
            sections: {
                search: true,
                folders: true,
                tags: true,
                filters: false,
                imageInfo: true,
                catalogInfo: false,
                quickActions: true,
                activeJobs: true,
                jobHistory: false
            },

            // Job streaming
            streamingJob: null,
            streamEvents: [],
            streamEventSource: null,
            streamConnectionStatus: null,
            streamConnectionStatusText: '',

            // Map/Time View
            mapInstance: null,
            mapMarkers: null,
            mapStats: { displayed: 0, totalPhotos: 0 },
            mapClustersCache: null, // Cache for cluster data
            mapLastPrecision: null, // Track last precision level
            mapLastDateRange: null, // Track last date range
            timelineSlider: null,
            timelineRange: { from: null, to: null, min: null, max: null },
            timelineData: null,
            timelineBuckets: [],
            selectedCluster: null,
            clusterImages: [],
            clusterImagesLoading: false,
            mapboxToken: 'pk.eyJ1IjoiaXJqdWRzb24iLCJhIjoiY2lnMTk4dzFuMHBhbnV3bHZsMmE0Ym1hcCJ9.LQSOcDk_TOrObpLYB-7_xw', // Mapbox token for better tiles

            // Duplicates View
            duplicateGroups: [],
            duplicatesTotal: 0,
            duplicatesLoading: false,
            duplicatesPage: 0,
            duplicatesPageSize: 20,
            duplicatesStats: null,
            duplicatesFilter: {
                reviewed: null,
                similarity_type: null
            },
            selectedDuplicateGroup: null,

            // Burst Detection
            bursts: [],
            currentBurst: null,
            showBurstModal: false,
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
                // Calculate duration from created/updated timestamps
                let durationSeconds = null;
                if (metadata?.created && metadata?.updated) {
                    const start = new Date(metadata.created);
                    const end = new Date(metadata.updated);
                    durationSeconds = Math.round((end - start) / 1000);
                }
                return {
                    id: jobId,
                    name: metadata?.name || 'Job',
                    started: metadata?.created || metadata?.started || null,
                    status: details.status || 'PENDING',
                    progress: details.progress || {},
                    result: details.result || {},
                    durationSeconds: durationSeconds,
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

        completedJobs() {
            return this.jobs.filter(job =>
                job.status === 'SUCCESS' || job.status === 'FAILURE'
            );
        },

        allJobsArchived() {
            if (this.completedJobs.length === 0) return false;
            return this.completedJobs.every(job => this.archivedJobs.has(job.id));
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
            // Clean up map view when leaving
            if (this.currentView === 'maptime' && view !== 'maptime') {
                this.destroyMapView();
            }

            this.currentView = view;

            if (view === 'jobs') {
                this.startJobsRefresh();
            } else if (view === 'browse') {
                if (this.images.length === 0) {
                    this.loadImages(true);
                }
            } else if (view === 'maptime') {
                this.$nextTick(() => this.initMapView());
            } else if (view === 'duplicates') {
                this.loadDuplicates(true);
            }
        },

        // Panel toggle for collapsible sections
        toggleSection(section) {
            this.sections[section] = !this.sections[section];
        },

        // Keyboard shortcut handler for panel toggles
        handlePanelShortcuts(e) {
            // Only handle in browse view
            if (this.currentView !== 'browse') return;

            // F6 - Toggle left panel
            if (e.key === 'F6') {
                e.preventDefault();
                this.showLeftPanel = !this.showLeftPanel;
            }
            // F7 - Toggle right panel
            else if (e.key === 'F7') {
                e.preventDefault();
                this.showRightPanel = !this.showRightPanel;
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
                    // Load stats for the current catalog
                    this.loadCatalogStats();
                }
            } catch (error) {
                console.error('Error loading catalogs:', error);
                this.addNotification('Failed to load catalogs', 'error');
            }
        },

        switchCatalog(catalog, showNotification = true) {
            this.currentCatalog = catalog;
            this.analyzeForm.catalog_id = catalog.id;

            // Only show notification if explicitly switching (not on first load or single catalog)
            if (showNotification && this.catalogs.length > 1) {
                this.addNotification(`Switched to catalog: ${catalog.name}`, 'success');
            }

            // Clear any active filters when switching catalogs
            this.clearFilters();

            // Load catalog stats, filter options, tags, bursts, and images
            this.loadCatalogStats();
            this.loadFilterOptions();
            this.loadAvailableTags();
            this.loadBursts();
            if (this.currentView === 'browse') {
                this.loadImages(true);
            }
        },

        async loadCatalogStats() {
            if (!this.currentCatalog) {
                this.catalogStats = null;
                return;
            }

            this.catalogStatsLoading = true;
            try {
                const response = await axios.get(`/api/catalogs/${this.currentCatalog.id}/stats`);
                this.catalogStats = response.data;
            } catch (error) {
                console.error('Error loading catalog stats:', error);
                this.catalogStats = null;
            } finally {
                this.catalogStatsLoading = false;
            }
        },

        async loadFilterOptions() {
            if (!this.currentCatalog) {
                this.filterOptions = null;
                return;
            }

            try {
                const response = await axios.get(`/api/catalogs/${this.currentCatalog.id}/filter-options`);
                this.filterOptions = response.data;
            } catch (error) {
                console.error('Error loading filter options:', error);
                this.filterOptions = null;
            }
        },

        async loadAvailableTags() {
            if (!this.currentCatalog) {
                this.availableTags = [];
                return;
            }

            this.tagsLoading = true;
            try {
                const response = await axios.get(`/api/catalogs/${this.currentCatalog.id}/tags`, {
                    params: { limit: 500, sort_by: 'count', sort_order: 'desc' }
                });
                this.availableTags = response.data.tags;
            } catch (error) {
                console.error('Error loading available tags:', error);
                this.availableTags = [];
            } finally {
                this.tagsLoading = false;
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
                        created: job.created_at,
                        updated: job.updated_at
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

                // Get previous data BEFORE updating cache (for status transition detection)
                const previousData = this.jobDetailsCache[jobId];
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

                // Auto-collapse completed jobs ONLY when they transition from in-progress to completed
                // This respects user's manual expand/collapse choices - once a job is completed,
                // subsequent polls won't force-collapse it again
                const wasInProgress = previousData && (previousData.status === 'PENDING' || previousData.status === 'PROGRESS');
                const isNowComplete = jobData.status === 'SUCCESS' || jobData.status === 'FAILURE';
                if (isNowComplete && wasInProgress) {
                    // Job just transitioned to completed state - auto-archive it
                    this.archivedJobs.add(jobId);
                    this.archivedJobs = new Set(this.archivedJobs);
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

                    // Auto-collapse completed jobs
                    this.archivedJobs.add(jobId);
                    this.archivedJobs = new Set(this.archivedJobs);

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

        async dismissJob(jobId) {
            // Revoke job on server (non-terminating) and remove from UI
            try {
                await axios.delete(`/api/jobs/${jobId}`, {
                    params: { terminate: false }
                });
            } catch (error) {
                // Ignore errors - job may already be gone
                console.log(`Could not revoke job ${jobId}:`, error.message);
            }
            // Always remove from UI
            this.removeJob(jobId);
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

        // Show confirmation modal for scan job
        startScan() {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog', 'error');
                return;
            }
            this.showScanConfirmModal = true;
        },

        // Show confirmation modal for auto-tag job
        startAutoTag() {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog', 'error');
                return;
            }
            this.showAutoTagConfirmModal = true;
        },

        // Show confirmation modal for find duplicates job
        startFindDuplicates() {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog', 'error');
                return;
            }
            this.showDuplicatesConfirmModal = true;
        },

        // Actually submit the scan job (called after confirmation)
        async confirmScan() {
            try {
                this.showScanConfirmModal = false;
                const continuePipeline = this.scanContinuePipeline;
                this.scanContinuePipeline = false; // Reset for next time

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
                        started: new Date().toISOString(),
                        continuePipeline: continuePipeline
                    };
                    this.persistJobs(); // Save to localStorage
                    this.loadJobDetails(response.data.job_id);

                    // If pipeline continuation is enabled, watch for job completion
                    if (continuePipeline) {
                        this.watchJobForPipeline(response.data.job_id, 'scan');
                    }
                }

                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit scan job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },

        // Actually submit the auto-tag job (called after confirmation)
        async confirmAutoTag() {
            try {
                this.showAutoTagConfirmModal = false;
                const continuePipeline = this.autoTagContinuePipeline;
                const backend = this.autoTagBackend;
                const tagMode = this.autoTagMode;
                this.autoTagContinuePipeline = false; // Reset for next time

                const response = await axios.post(
                    `/api/catalogs/${this.currentCatalog.id}/auto-tag`,
                    null,
                    { params: { backend: backend, tag_mode: tagMode, continue_pipeline: continuePipeline } }
                );

                this.addNotification('Auto-tagging job submitted successfully', 'success');

                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    // Store job metadata
                    const modeLabel = tagMode === 'all' ? 'retag all' : 'untagged only';
                    this.jobMetadata[response.data.job_id] = {
                        name: `Auto-Tag (${backend}, ${modeLabel}): ${this.currentCatalog.name}`,
                        started: new Date().toISOString()
                    };
                    this.persistJobs(); // Save to localStorage
                    this.loadJobDetails(response.data.job_id);
                }

                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit auto-tag job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },

        // Actually submit the find duplicates job (called after confirmation)
        async confirmFindDuplicates() {
            try {
                this.showDuplicatesConfirmModal = false;

                // Use the catalogs API for duplicate detection
                const response = await axios.post(
                    `/api/catalogs/${this.currentCatalog.id}/detect-duplicates`,
                    null,
                    { params: { similarity_threshold: 5, recompute_hashes: false } }
                );

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

        // Watch a job and continue pipeline when it completes
        watchJobForPipeline(jobId, jobType) {
            const checkInterval = setInterval(async () => {
                try {
                    const response = await axios.get(`/api/jobs/${jobId}`);
                    const status = response.data.status;

                    if (status === 'SUCCESS' || status === 'success') {
                        clearInterval(checkInterval);

                        // Continue to next step based on job type
                        if (jobType === 'scan') {
                            this.addNotification('Scan complete, starting Auto-Tag...', 'info');
                            // Start auto-tag with continue_pipeline=true to chain to duplicates
                            await axios.post(
                                `/api/catalogs/${this.currentCatalog.id}/auto-tag`,
                                null,
                                { params: { backend: 'openclip', continue_pipeline: true } }
                            ).then(res => {
                                if (res.data.job_id) {
                                    this.allJobs.unshift(res.data.job_id);
                                    this.jobMetadata[res.data.job_id] = {
                                        name: `Auto-Tag (pipeline): ${this.currentCatalog.name}`,
                                        started: new Date().toISOString()
                                    };
                                    this.persistJobs();
                                    this.loadJobDetails(res.data.job_id);
                                }
                            });
                        }
                    } else if (status === 'FAILURE' || status === 'failure') {
                        clearInterval(checkInterval);
                        this.addNotification('Pipeline stopped: previous job failed', 'error');
                    }
                } catch (e) {
                    console.error('Error checking job status for pipeline:', e);
                }
            }, 5000); // Check every 5 seconds

            // Stop checking after 2 hours max
            setTimeout(() => clearInterval(checkInterval), 2 * 60 * 60 * 1000);
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

        getJobIcon(name) {
            if (!name) return '⚙️';
            const nameLower = name.toLowerCase();
            if (nameLower.includes('scan')) return '📂';
            if (nameLower.includes('analyze')) return '🔍';
            if (nameLower.includes('tag') || nameLower.includes('auto-tag')) return '🏷️';
            if (nameLower.includes('duplicate')) return '🔄';
            return '⚙️';
        },

        isJobArchived(jobId) {
            return this.archivedJobs.has(jobId);
        },

        toggleJobArchived(jobId) {
            if (this.archivedJobs.has(jobId)) {
                this.archivedJobs.delete(jobId);
            } else {
                this.archivedJobs.add(jobId);
            }
            // Force reactivity update
            this.archivedJobs = new Set(this.archivedJobs);
        },

        toggleAllArchived() {
            if (this.allJobsArchived) {
                // Expand all
                this.archivedJobs.clear();
            } else {
                // Collapse all completed jobs
                this.completedJobs.forEach(job => {
                    this.archivedJobs.add(job.id);
                });
            }
            // Force reactivity update
            this.archivedJobs = new Set(this.archivedJobs);
        },

        // Job control methods
        async cancelJob(jobId) {
            // Cancel job (soft revoke - marks as REVOKED in DB)
            await this.revokeJob(jobId, false);
        },

        async killJob(jobId) {
            // Kill job (terminate + force delete from DB)
            if (!confirm('Force kill this job? This will terminate it and remove it from the database.')) {
                return;
            }

            try {
                await axios.delete(`/api/jobs/${jobId}`, {
                    params: { terminate: true, force: true }
                });

                // Remove from tracked jobs
                this.removeJob(jobId);

                const jobName = this.jobMetadata[jobId]?.name || jobId.substring(0, 8);
                this.addNotification(`Job ${jobName} killed and removed`, 'success');
            } catch (error) {
                console.error(`Failed to kill job ${jobId}:`, error);
                this.addNotification('Failed to kill job', 'error');
            }
        },

        rerunJob(jobId) {
            this.addNotification('Job rerun not yet implemented', 'info');
        },

        showJobDetails(job) {
            // Show full job details in a modal/alert for now
            const details = {
                id: job.id,
                name: job.name,
                status: job.status,
                started: job.started,
                result: job.result
            };
            console.log('Job details:', details);
            // For now, show as an alert - could be enhanced with a proper modal later
            alert(JSON.stringify(details, null, 2));
        },

        async deleteJob(jobId) {
            if (!confirm('Delete this job from history?')) {
                return;
            }

            try {
                // Delete with force=true to remove from database
                await axios.delete(`/api/jobs/${jobId}`, {
                    params: { force: true }
                });

                // Remove from tracked jobs
                this.removeJob(jobId);

                this.addNotification('Job deleted from history', 'success');
            } catch (error) {
                console.error(`Failed to delete job ${jobId}:`, error);
                this.addNotification('Failed to delete job', 'error');
            }
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

                // Build params with filters and sort
                const params = {
                    limit: this.imagesPageSize,
                    offset: offset,
                    sort_by: this.sortBy,
                    sort_order: this.sortOrder
                };

                // Add search filter
                if (this.filters.search) {
                    params.search = this.filters.search;
                }

                // Add other filters
                if (this.filters.file_type) {
                    params.file_type = this.filters.file_type;
                }
                if (this.filters.camera_make) {
                    params.camera_make = this.filters.camera_make;
                }
                if (this.filters.camera_model) {
                    params.camera_model = this.filters.camera_model;
                }
                if (this.filters.lens) {
                    params.lens = this.filters.lens;
                }
                if (this.filters.focal_length) {
                    params.focal_length = this.filters.focal_length;
                }
                if (this.filters.f_stop) {
                    params.f_stop = this.filters.f_stop;
                }
                if (this.filters.has_gps !== null && this.filters.has_gps !== '') {
                    params.has_gps = this.filters.has_gps;
                }
                if (this.filters.date_from) {
                    params.date_from = this.filters.date_from;
                }
                if (this.filters.date_to) {
                    params.date_to = this.filters.date_to;
                }

                // Add tag filters
                if (this.filters.tags) {
                    params.tags = this.filters.tags;
                    params.tag_match = this.filters.tag_match;
                }
                if (this.filters.has_tags !== null && this.filters.has_tags !== '') {
                    params.has_tags = this.filters.has_tags;
                }

                // Always include tags in response for display
                params.include_tags = true;

                const response = await axios.get(`/api/catalogs/${this.currentCatalog.id}/images`, {
                    params: params
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

        // Search and filter methods
        onSearchInput() {
            // Debounce search input
            if (this.searchDebounceTimer) {
                clearTimeout(this.searchDebounceTimer);
            }
            this.searchDebounceTimer = setTimeout(() => {
                this.loadImages(true);
            }, 300);
        },

        applyFilters() {
            this.loadImages(true);
        },

        clearFilters() {
            this.filters = {
                search: '',
                file_type: '',
                camera_make: '',
                camera_model: '',
                lens: '',
                focal_length: '',
                f_stop: '',
                has_gps: null,
                date_from: '',
                date_to: '',
                tags: '',
                tag_match: 'any',
                has_tags: null
            };
            this.sortBy = 'date';
            this.sortOrder = 'desc';
            this.loadImages(true);
        },

        hasActiveFilters() {
            return this.filters.search ||
                   this.filters.file_type ||
                   this.filters.camera_make ||
                   this.filters.camera_model ||
                   this.filters.lens ||
                   this.filters.focal_length ||
                   this.filters.f_stop ||
                   this.filters.has_gps !== null ||
                   this.filters.date_from ||
                   this.filters.date_to ||
                   this.filters.tags ||
                   this.filters.has_tags !== null;
        },

        // Tag filter helpers
        toggleTagFilter(tagName) {
            const tags = this.filters.tags ? this.filters.tags.split(',').map(t => t.trim()).filter(t => t) : [];
            const index = tags.indexOf(tagName);

            if (index >= 0) {
                tags.splice(index, 1);
            } else {
                tags.push(tagName);
            }

            this.filters.tags = tags.join(',');
            this.applyFilters();
        },

        isTagSelected(tagName) {
            if (!this.filters.tags) return false;
            const tags = this.filters.tags.split(',').map(t => t.trim());
            return tags.includes(tagName);
        },

        onSortChange() {
            this.loadImages(true);
        },

        // Semantic Search methods
        async performSemanticSearch() {
            if (!this.semanticSearchQuery || !this.currentCatalog) return;

            this.isSearching = true;
            try {
                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/search`,
                    {
                        params: {
                            q: this.semanticSearchQuery,
                            limit: 100,
                            threshold: 0.2
                        }
                    }
                );

                this.searchResults = response.data.results;
                this.searchMode = true;

                // Update the image grid to show search results
                this.images = this.searchResults.map(r => ({
                    id: r.image_id,
                    source_path: r.source_path,
                    similarity_score: r.similarity_score,
                }));
                this.imagesTotal = this.searchResults.length;

                this.addNotification(`Found ${this.searchResults.length} images`, 'success');
            } catch (error) {
                console.error('Search failed:', error);
                this.addNotification('Search failed: ' + (error.response?.data?.detail || error.message), 'error');
            } finally {
                this.isSearching = false;
            }
        },

        clearSearch() {
            this.searchQuery = '';
            this.semanticSearchQuery = '';
            this.filters.search = '';
            this.searchResults = [];
            this.searchMode = false;
            this.lastSearchType = null;
            this.loadImages(true);
        },

        // Detect if query is semantic (descriptive phrase) or text (filename)
        isSemanticQuery(query) {
            if (!query) return false;
            const trimmed = query.trim();
            const words = trimmed.split(/\s+/);

            // Heuristics for semantic search:
            // 1. 3+ words is likely descriptive
            // 2. Contains common descriptive words
            // 3. Not a filename pattern (no extension, no path separators)

            const hasExtension = /\.\w{2,4}$/.test(trimmed);
            const hasPath = trimmed.includes('/') || trimmed.includes('\\');
            const descriptiveWords = ['with', 'of', 'in', 'at', 'the', 'a', 'an', 'on', 'by', 'for', 'showing', 'featuring'];
            const hasDescriptiveWord = descriptiveWords.some(w => words.map(w => w.toLowerCase()).includes(w));

            if (hasExtension || hasPath) return false;
            if (words.length >= 3) return true;
            if (hasDescriptiveWord) return true;

            return false;
        },

        getSearchModeHint() {
            if (!this.searchQuery) return 'Enter filename or description';
            if (this.isSemanticQuery(this.searchQuery)) {
                return 'semantic search';
            }
            return 'text search';
        },

        onSearchQueryChange() {
            // For short text queries, do instant filtering
            if (this.searchQuery && !this.isSemanticQuery(this.searchQuery)) {
                if (this.searchDebounceTimer) {
                    clearTimeout(this.searchDebounceTimer);
                }
                this.searchDebounceTimer = setTimeout(() => {
                    this.filters.search = this.searchQuery;
                    this.lastSearchType = 'text';
                    this.searchMode = true;
                    this.loadImages(true);
                }, 300);
            }
        },

        async performUnifiedSearch() {
            if (!this.searchQuery || !this.currentCatalog) return;

            if (this.isSemanticQuery(this.searchQuery)) {
                // Perform semantic search
                this.isSearching = true;
                this.lastSearchType = 'semantic';
                try {
                    const response = await axios.get(
                        `/api/catalogs/${this.currentCatalog.id}/search`,
                        {
                            params: {
                                q: this.searchQuery,
                                limit: 100,
                                threshold: 0.2
                            }
                        }
                    );

                    this.searchResults = response.data.results;
                    this.searchMode = true;
                    this.filters.search = ''; // Clear text filter

                    this.images = this.searchResults.map(r => ({
                        id: r.image_id,
                        source_path: r.source_path,
                        similarity_score: r.similarity_score,
                    }));
                    this.imagesTotal = this.searchResults.length;

                    this.addNotification(`Found ${this.searchResults.length} images`, 'success');
                } catch (error) {
                    console.error('Semantic search failed:', error);
                    this.addNotification('Search failed: ' + (error.response?.data?.detail || error.message), 'error');
                } finally {
                    this.isSearching = false;
                }
            } else {
                // Text search - apply immediately
                this.filters.search = this.searchQuery;
                this.lastSearchType = 'text';
                this.searchMode = true;
                this.searchResults = [];
                this.loadImages(true);
            }
        },

        async findSimilarImages(imageId) {
            if (!this.currentCatalog || !imageId) return;

            try {
                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/similar/${imageId}`,
                    {
                        params: { limit: 50, threshold: 0.5 }
                    }
                );

                this.searchResults = response.data.results;
                this.searchMode = true;
                this.semanticSearchQuery = `Similar to image`;

                this.images = this.searchResults.map(r => ({
                    id: r.image_id,
                    source_path: r.source_path,
                    similarity_score: r.similarity_score,
                }));
                this.imagesTotal = this.searchResults.length;

                this.addNotification(`Found ${this.searchResults.length} similar images`, 'success');
            } catch (error) {
                console.error('Find similar failed:', error);
                this.addNotification('Find similar failed: ' + (error.response?.data?.detail || error.message), 'error');
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

        handleScroll(event) {
            if (!this.infiniteScrollEnabled || this.imagesLoading || !this.hasMoreImages) {
                return;
            }

            // Check if we're in browse view
            if (this.currentView !== 'browse') {
                return;
            }

            // Get the scroll container (.main-content element)
            const container = event ? event.target : document.querySelector('.main-content');
            if (!container) return;

            const scrollTop = container.scrollTop;
            const scrollHeight = container.scrollHeight;
            const clientHeight = container.clientHeight;

            // Load more when within threshold of bottom
            if (scrollHeight - scrollTop - clientHeight < this.scrollThreshold) {
                this.loadImages(false);
            }
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

        // =====================================================================
        // Map/Time View Methods
        // =====================================================================

        async initMapView() {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog first', 'info');
                return;
            }

            // Initialize map if not already created
            if (!this.mapInstance) {
                const mapElement = document.getElementById('photo-map');
                if (!mapElement) {
                    console.error('Map element not found');
                    return;
                }

                this.mapInstance = L.map('photo-map', {
                    center: [39.8283, -98.5795], // Center of US
                    zoom: 4
                });

                // Use Mapbox tiles if token provided, otherwise OpenStreetMap
                if (this.mapboxToken) {
                    L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}', {
                        id: 'mapbox/streets-v12',
                        accessToken: this.mapboxToken,
                        tileSize: 512,
                        zoomOffset: -1,
                        attribution: '&copy; <a href="https://www.mapbox.com/">Mapbox</a>'
                    }).addTo(this.mapInstance);
                } else {
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    }).addTo(this.mapInstance);
                }

                // Add event listeners for map movement
                // Only update clusters when zoom changes (which affects precision)
                // Panning within the same precision level uses cached data
                this.mapInstance.on('zoomend', () => this.updateMapClusters());
                this.mapInstance.on('moveend', () => this.renderCachedClusters());
            }

            // Load timeline data and initialize slider
            await this.loadTimelineData();
            this.initTimelineSlider();
            await this.updateMapClusters();
        },

        async loadTimelineData() {
            try {
                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/map/timeline`,
                    { params: { bucket_size: 'month' } }
                );

                this.timelineData = response.data;
                this.timelineBuckets = response.data.buckets;

                if (response.data.date_range.min && response.data.date_range.max) {
                    this.timelineRange.min = new Date(response.data.date_range.min);
                    this.timelineRange.max = new Date(response.data.date_range.max);
                    this.timelineRange.from = this.timelineRange.min;
                    this.timelineRange.to = this.timelineRange.max;
                }

                this.drawHistogram();
            } catch (error) {
                console.error('Error loading timeline data:', error);
                this.addNotification('Failed to load timeline data', 'error');
            }
        },

        drawHistogram() {
            const canvas = this.$refs.histogramCanvas;
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            const container = this.$refs.histogramContainer;
            if (!container) return;

            // Set canvas size to match container
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!this.timelineBuckets.length) return;

            const maxCount = Math.max(...this.timelineBuckets.map(b => b.count));
            if (maxCount === 0) return;

            const barWidth = canvas.width / this.timelineBuckets.length;
            const padding = 2;

            this.timelineBuckets.forEach((bucket, index) => {
                const bucketDate = new Date(bucket.date);
                const inRange = this.timelineRange.from && this.timelineRange.to &&
                    bucketDate >= this.timelineRange.from && bucketDate <= this.timelineRange.to;

                // Draw total count bar (gray, behind)
                const totalHeight = (bucket.count / maxCount) * (canvas.height - 4);
                const x = index * barWidth + 1;
                const y = canvas.height - totalHeight - 2;

                ctx.fillStyle = inRange ? 'rgba(100, 116, 139, 0.4)' : 'rgba(100, 116, 139, 0.2)';
                ctx.fillRect(x, y, barWidth - padding, totalHeight);

                // Draw GPS count bar (blue, in front)
                const gpsHeight = (bucket.count_with_gps / maxCount) * (canvas.height - 4);
                const gpsY = canvas.height - gpsHeight - 2;

                ctx.fillStyle = inRange ? 'rgba(96, 165, 250, 0.9)' : 'rgba(96, 165, 250, 0.3)';
                ctx.fillRect(x, gpsY, barWidth - padding, gpsHeight);
            });
        },

        initTimelineSlider() {
            const sliderElement = document.getElementById('timeline-slider');
            if (!sliderElement || this.timelineSlider) return;

            if (!this.timelineRange.min || !this.timelineRange.max) return;

            const minTime = this.timelineRange.min.getTime();
            const maxTime = this.timelineRange.max.getTime();

            if (minTime >= maxTime) return;

            this.timelineSlider = noUiSlider.create(sliderElement, {
                start: [minTime, maxTime],
                connect: true,
                range: {
                    'min': minTime,
                    'max': maxTime
                },
                step: 24 * 60 * 60 * 1000 // 1 day
            });

            this.timelineSlider.on('slide', (values) => {
                this.timelineRange.from = new Date(parseInt(values[0]));
                this.timelineRange.to = new Date(parseInt(values[1]));
                this.drawHistogram();
            });

            this.timelineSlider.on('change', () => {
                this.updateMapClusters();
            });
        },

        async updateMapClusters(forceRefresh = false) {
            if (!this.mapInstance || !this.currentCatalog) return;

            const zoom = this.mapInstance.getZoom();

            // Map zoom to precision
            let precision = 4;
            if (zoom <= 4) precision = 2;
            else if (zoom <= 8) precision = 4;
            else if (zoom <= 12) precision = 6;
            else precision = 8;

            // Build current date range key for cache comparison
            const dateRangeKey = this.timelineRange.from && this.timelineRange.to
                ? `${this.timelineRange.from.getTime()}-${this.timelineRange.to.getTime()}`
                : 'all';

            // Check if we can use cached data (same precision and date range)
            if (!forceRefresh &&
                this.mapClustersCache &&
                this.mapLastPrecision === precision &&
                this.mapLastDateRange === dateRangeKey) {
                // Use cached data, just re-render visible clusters
                this.renderCachedClusters();
                return;
            }

            try {
                // Query ALL clusters for this precision/date range (no bounds filter)
                // This way we cache all data and can pan freely without re-querying
                const params = { precision };

                if (this.timelineRange.from) {
                    params.date_from = this.timelineRange.from.toISOString();
                }
                if (this.timelineRange.to) {
                    params.date_to = this.timelineRange.to.toISOString();
                }

                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/map/clusters`,
                    { params }
                );

                // Cache the cluster data
                this.mapClustersCache = response.data.clusters;
                this.mapLastPrecision = precision;
                this.mapLastDateRange = dateRangeKey;
                this.mapStats.totalPhotos = response.data.total_with_gps;

                // Render clusters visible in current bounds
                this.renderCachedClusters();
            } catch (error) {
                console.error('Error loading map clusters:', error);
            }
        },

        renderCachedClusters() {
            if (!this.mapInstance || !this.mapClustersCache) return;

            const bounds = this.mapInstance.getBounds();

            // Filter clusters to those visible in current bounds
            const visibleClusters = this.mapClustersCache.filter(cluster => {
                return bounds.contains([cluster.center_lat, cluster.center_lon]);
            });

            this.mapStats.displayed = visibleClusters.length;

            // Clear existing markers
            if (this.mapMarkers) {
                this.mapInstance.removeLayer(this.mapMarkers);
            }
            this.mapMarkers = L.layerGroup();

            // Add cluster markers for visible clusters only
            visibleClusters.forEach(cluster => {
                const marker = this.createClusterMarker(cluster);
                this.mapMarkers.addLayer(marker);
            });

            this.mapMarkers.addTo(this.mapInstance);
        },

        createClusterMarker(cluster) {
            // Determine size class based on count
            let sizeClass = 'marker-cluster-small';
            let size = 40;

            if (cluster.count > 100) {
                sizeClass = 'marker-cluster-large';
                size = 50;
            } else if (cluster.count > 20) {
                sizeClass = 'marker-cluster-medium';
                size = 45;
            }

            // Format count for display
            const displayCount = cluster.count >= 1000
                ? Math.round(cluster.count / 1000) + 'K'
                : cluster.count;

            const icon = L.divIcon({
                html: `<div>${displayCount}</div>`,
                className: `marker-cluster ${sizeClass}`,
                iconSize: L.point(size, size)
            });

            const marker = L.marker([cluster.center_lat, cluster.center_lon], { icon });

            // Single click: zoom in if not at max zoom
            marker.on('click', () => {
                if (this.mapInstance.getZoom() < 16) {
                    this.mapInstance.setView(
                        [cluster.center_lat, cluster.center_lon],
                        this.mapInstance.getZoom() + 3
                    );
                } else {
                    this.showClusterImages(cluster);
                }
            });

            // Double click: show cluster images
            marker.on('dblclick', (e) => {
                L.DomEvent.stopPropagation(e);
                this.showClusterImages(cluster);
            });

            return marker;
        },

        async showClusterImages(cluster) {
            this.selectedCluster = cluster;
            this.clusterImages = [];
            this.clusterImagesLoading = true;

            try {
                const params = {
                    geohash: cluster.geohash,
                    limit: 30
                };

                if (this.timelineRange.from) {
                    params.date_from = this.timelineRange.from.toISOString();
                }
                if (this.timelineRange.to) {
                    params.date_to = this.timelineRange.to.toISOString();
                }

                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/map/images`,
                    { params }
                );

                this.clusterImages = response.data.images;
            } catch (error) {
                console.error('Error loading cluster images:', error);
                this.addNotification('Failed to load cluster images', 'error');
            } finally {
                this.clusterImagesLoading = false;
            }
        },

        async loadMoreClusterImages() {
            if (!this.selectedCluster || this.clusterImagesLoading) return;

            this.clusterImagesLoading = true;

            try {
                const params = {
                    geohash: this.selectedCluster.geohash,
                    limit: 30,
                    offset: this.clusterImages.length
                };

                if (this.timelineRange.from) {
                    params.date_from = this.timelineRange.from.toISOString();
                }
                if (this.timelineRange.to) {
                    params.date_to = this.timelineRange.to.toISOString();
                }

                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/map/images`,
                    { params }
                );

                this.clusterImages.push(...response.data.images);
            } catch (error) {
                console.error('Error loading more cluster images:', error);
            } finally {
                this.clusterImagesLoading = false;
            }
        },

        getClusterThumbnailUrl(image) {
            if (image.thumbnail_path) {
                return `/api/catalogs/${this.currentCatalog.id}/thumbnails/${image.thumbnail_path}`;
            }
            return `/api/catalogs/${this.currentCatalog.id}/images/${image.id}/thumbnail?size=small`;
        },

        openClusterLightbox(image) {
            // Find image in main images array or create temporary lightbox
            const index = this.images.findIndex(img => img.id === image.id);
            if (index >= 0) {
                this.openLightbox(index);
            } else {
                // Open directly with this image
                this.lightboxImage = image;
                this.lightboxVisible = true;
            }
        },

        formatDateShort(date) {
            if (!date) return '—';
            return new Date(date).toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'short'
            });
        },

        destroyMapView() {
            if (this.mapInstance) {
                this.mapInstance.remove();
                this.mapInstance = null;
            }
            if (this.timelineSlider) {
                this.timelineSlider.destroy();
                this.timelineSlider = null;
            }
            this.mapMarkers = null;
            this.mapClustersCache = null;
            this.mapLastPrecision = null;
            this.mapLastDateRange = null;
            this.selectedCluster = null;
            this.clusterImages = [];
            this.timelineData = null;
            this.timelineBuckets = [];
        },

        // =====================================================================
        // Duplicates View Methods
        // =====================================================================

        async loadDuplicates(reset = false) {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog first', 'info');
                return;
            }

            if (reset) {
                this.duplicateGroups = [];
                this.duplicatesPage = 0;
                this.duplicatesTotal = 0;
            }

            this.duplicatesLoading = true;

            try {
                const offset = this.duplicatesPage * this.duplicatesPageSize;

                const params = {
                    limit: this.duplicatesPageSize,
                    offset: offset
                };

                // Add filters
                if (this.duplicatesFilter.reviewed !== null) {
                    params.reviewed = this.duplicatesFilter.reviewed;
                }
                if (this.duplicatesFilter.similarity_type) {
                    params.similarity_type = this.duplicatesFilter.similarity_type;
                }

                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/duplicates`,
                    { params }
                );

                if (reset) {
                    this.duplicateGroups = response.data.groups;
                } else {
                    this.duplicateGroups.push(...response.data.groups);
                }

                this.duplicatesTotal = response.data.total;
                this.duplicatesStats = response.data.statistics;
                this.duplicatesPage++;
            } catch (error) {
                console.error('Error loading duplicates:', error);
                this.addNotification('Failed to load duplicates', 'error');
            } finally {
                this.duplicatesLoading = false;
            }
        },

        async loadDuplicatesStats() {
            if (!this.currentCatalog) return;

            try {
                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/duplicates/stats`
                );
                this.duplicatesStats = response.data;
            } catch (error) {
                console.error('Error loading duplicate stats:', error);
            }
        },

        selectDuplicateGroup(group) {
            this.selectedDuplicateGroup = group;
        },

        closeDuplicateGroup() {
            this.selectedDuplicateGroup = null;
        },

        getDuplicateThumbnailUrl(member) {
            if (!this.currentCatalog) return '';
            return `/api/catalogs/${this.currentCatalog.id}/images/${member.image_id}/thumbnail?size=medium`;
        },

        getSimilarityLabel(type) {
            return type === 'exact' ? 'Exact Match' : 'Perceptual Match';
        },

        getSimilarityClass(type) {
            return type === 'exact' ? 'badge-exact' : 'badge-perceptual';
        },

        hasMoreDuplicates() {
            return this.duplicateGroups.length < this.duplicatesTotal;
        },

        applyDuplicatesFilter() {
            this.loadDuplicates(true);
        },

        clearDuplicatesFilter() {
            this.duplicatesFilter = {
                reviewed: null,
                similarity_type: null
            };
            this.loadDuplicates(true);
        },

        // =====================================================================
        // Burst Detection Methods
        // =====================================================================

        async loadBursts() {
            if (!this.currentCatalog) return;

            try {
                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/bursts`
                );
                this.bursts = response.data.bursts;
            } catch (error) {
                console.error('Failed to load bursts:', error);
            }
        },

        async viewBurst(burstId) {
            try {
                const response = await axios.get(
                    `/api/catalogs/${this.currentCatalog.id}/bursts/${burstId}`
                );
                this.currentBurst = response.data;
                this.showBurstModal = true;
            } catch (error) {
                console.error('Failed to load burst:', error);
                this.addNotification('Failed to load burst details', 'error');
            }
        },

        async setBestImage(burstId, imageId) {
            try {
                await axios.put(
                    `/api/catalogs/${this.currentCatalog.id}/bursts/${burstId}`,
                    { best_image_id: imageId }
                );
                this.addNotification('Best image updated', 'success');

                // Refresh current burst if modal is open
                if (this.showBurstModal && this.currentBurst && this.currentBurst.id === burstId) {
                    await this.viewBurst(burstId);
                }

                // Reload bursts list
                await this.loadBursts();
            } catch (error) {
                console.error('Failed to update burst:', error);
                this.addNotification('Failed to update best image', 'error');
            }
        },

        async startBurstDetection() {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog first', 'error');
                return;
            }

            try {
                const response = await axios.post(
                    `/api/catalogs/${this.currentCatalog.id}/detect-bursts`
                );

                this.addNotification('Burst detection started', 'success');

                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    this.jobMetadata[response.data.job_id] = {
                        name: `Detect Bursts: ${this.currentCatalog.name}`,
                        started: new Date().toISOString()
                    };
                    this.persistJobs();
                    this.loadJobDetails(response.data.job_id);
                }

                this.setView('jobs');
            } catch (error) {
                console.error('Failed to start burst detection:', error);
                this.addNotification('Failed to start burst detection: ' + (error.response?.data?.detail || error.message), 'error');
            }
        },

        async startGenerateThumbnails() {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog first', 'error');
                return;
            }

            try {
                const response = await axios.post('/api/jobs/start', {
                    job_type: 'generate_thumbnails',
                    catalog_id: this.currentCatalog.id
                });

                this.addNotification('Thumbnail generation started', 'success');

                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    this.jobMetadata[response.data.job_id] = {
                        name: `Generate Thumbnails: ${this.currentCatalog.name}`,
                        started: new Date().toISOString()
                    };
                    this.persistJobs();
                    this.loadJobDetails(response.data.job_id);
                }
            } catch (error) {
                console.error('Failed to start thumbnail generation:', error);
                this.addNotification('Failed to start thumbnail generation: ' + (error.response?.data?.detail || error.message), 'error');
            }
        },

        async startAnalyzeCatalog() {
            if (!this.currentCatalog) {
                this.addNotification('Please select a catalog first', 'error');
                return;
            }

            try {
                const response = await axios.post('/api/jobs/analyze', {
                    catalog_id: this.currentCatalog.id
                });

                this.addNotification('Catalog analysis started', 'success');

                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    this.jobMetadata[response.data.job_id] = {
                        name: `Analyze: ${this.currentCatalog.name}`,
                        started: new Date().toISOString()
                    };
                    this.persistJobs();
                    this.loadJobDetails(response.data.job_id);
                }
            } catch (error) {
                console.error('Failed to start catalog analysis:', error);
                this.addNotification('Failed to start catalog analysis: ' + (error.response?.data?.detail || error.message), 'error');
            }
        },

        getBurstInfo(image) {
            if (!image.burst_id) return null;
            const burst = this.bursts.find(b => b.id === image.burst_id);
            return burst;
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

        formatBytes(bytes) {
            if (!bytes || bytes === 0) return '0 B';
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(1024));
            return (bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
        },

        formatDuration(seconds) {
            if (!seconds || seconds === 0) return '0s';
            if (seconds < 60) return `${Math.round(seconds)}s`;
            if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
            return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
        },

        formatDate(dateStr) {
            if (!dateStr) return 'Unknown';
            return new Date(dateStr).toLocaleString();
        },

        getYear(dateStr) {
            if (!dateStr) return '?';
            try {
                const year = new Date(dateStr).getFullYear();
                // Validate year is reasonable (1800-2100)
                if (year >= 1800 && year <= 2100) {
                    return year;
                }
                return '?';
            } catch (e) {
                return '?';
            }
        },

        isValidDateRange(dateRange) {
            if (!dateRange || !dateRange.earliest || !dateRange.latest) return false;
            try {
                const earliest = new Date(dateRange.earliest).getFullYear();
                const latest = new Date(dateRange.latest).getFullYear();
                // Both years must be reasonable (1800-2100)
                return earliest >= 1800 && earliest <= 2100 &&
                       latest >= 1800 && latest <= 2100;
            } catch (e) {
                return false;
            }
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
            // Auto-select first catalog if available (no notification on first load)
            if (this.catalogs.length > 0) {
                this.switchCatalog(this.catalogs[0], false);
            }

            // Start monitoring worker health
            this.startJobsRefresh();
        });

        // Set up infinite scroll listener on .main-content element
        this._boundScrollHandler = this.handleScroll.bind(this);
        this.$nextTick(() => {
            this._scrollContainer = document.querySelector('.main-content');
            if (this._scrollContainer) {
                this._scrollContainer.addEventListener('scroll', this._boundScrollHandler, { passive: true });
            }
        });

        // Set up keyboard shortcuts for panel toggles (F6, F7)
        this._boundPanelShortcuts = this.handlePanelShortcuts.bind(this);
        window.addEventListener('keydown', this._boundPanelShortcuts);
    },

    beforeUnmount() {
        this.stopJobsRefresh();

        // Close all WebSocket connections
        Object.keys(this.jobWebSockets).forEach(jobId => {
            this.disconnectJobWebSocket(jobId);
        });

        // Remove scroll listener
        if (this._boundScrollHandler && this._scrollContainer) {
            this._scrollContainer.removeEventListener('scroll', this._boundScrollHandler);
        }

        // Remove keyboard shortcuts listener
        if (this._boundPanelShortcuts) {
            window.removeEventListener('keydown', this._boundPanelShortcuts);
        }
    }
}).mount('#app');
