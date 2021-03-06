// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/testkit/testapp.h>

#include <tests/proton/common/dummydbowner.h>
#include <vespa/config-imported-fields.h>
#include <vespa/config-rank-profiles.h>
#include <vespa/config-summarymap.h>
#include <vespa/fastos/file.h>
#include <vespa/document/datatype/documenttype.h>
#include <vespa/document/fieldvalue/intfieldvalue.h>
#include <vespa/document/repo/configbuilder.h>
#include <vespa/document/repo/documenttyperepo.h>
#include <vespa/document/repo/document_type_repo_factory.h>
#include <vespa/document/test/make_bucket_space.h>
#include <vespa/document/update/assignvalueupdate.h>
#include <vespa/document/update/documentupdate.h>
#include <vespa/searchcommon/common/schemaconfigurer.h>
#include <vespa/searchcore/proton/common/hw_info.h>
#include <vespa/searchcore/proton/matching/querylimiter.h>
#include <vespa/searchcore/proton/metrics/metricswireservice.h>
#include <vespa/searchcore/proton/persistenceengine/ipersistenceengineowner.h>
#include <vespa/searchcore/proton/persistenceengine/persistenceengine.h>
#include <vespa/searchcore/proton/server/bootstrapconfig.h>
#include <vespa/searchcore/proton/server/document_db_maintenance_config.h>
#include <vespa/searchcore/proton/server/documentdb.h>
#include <vespa/searchcore/proton/server/documentdbconfigmanager.h>
#include <vespa/searchcore/proton/server/fileconfigmanager.h>
#include <vespa/searchcore/proton/server/memoryconfigstore.h>
#include <vespa/searchcore/proton/server/persistencehandlerproxy.h>
#include <vespa/searchlib/index/dummyfileheadercontext.h>
#include <vespa/searchlib/transactionlog/translogserver.h>
#include <vespa/searchsummary/config/config-juniperrc.h>
#include <vespa/vespalib/util/lambdatask.h>
#include <vespa/config-bucketspaces.h>
#include <vespa/config-attributes.h>
#include <vespa/config-indexschema.h>
#include <vespa/config-summary.h>
#include <vespa/vespalib/io/fileutil.h>
#include <vespa/fastos/app.h>
#include <vespa/storage/bucketdb/config-stor-bucket-init.h>
#include <vespa/storage/config/config-stor-bouncer.h>
#include <vespa/storage/config/config-stor-communicationmanager.h>
#include <vespa/storage/config/config-stor-opslogger.h>
#include <vespa/storage/config/config-stor-prioritymapping.h>
#include <vespa/storage/config/config-stor-server.h>
#include <vespa/storage/config/config-stor-status.h>
#include <vespa/storage/visiting/config-stor-visitor.h>
#include <vespa/config-load-type.h>
#include <vespa/config-persistence.h>
#include <vespa/config-stor-distribution.h>
#include <vespa/config-stor-filestor.h>
#include <vespa/config-upgrading.h>
#include <vespa/config-slobroks.h>
#include <vespa/metrics/config-metricsmanager.h>
#include <vespa/storageserver/app/servicelayerprocess.h>
#include <vespa/storage/storageserver/storagenode.h>
#include <vespa/messagebus/config-messagebus.h>
#include <vespa/messagebus/testlib/slobrok.h>
#include <vespa/storage/storageserver/message_dispatcher.h>
#include <vespa/storage/storageserver/rpc/caching_rpc_target_resolver.h>
#include <vespa/storage/storageserver/rpc/message_codec_provider.h>
#include <vespa/storage/storageserver/rpc/shared_rpc_resources.h>
#include <vespa/storage/storageserver/rpc/slime_cluster_state_bundle_codec.h>
#include <vespa/storage/storageserver/rpc/storage_api_rpc_service.h>
#include <vespa/documentapi/loadtypes/loadtypeset.h>
#include <vespa/storageapi/messageapi/storagemessage.h>
#include <vespa/storageapi/message/persistence.h>
#include <vespa/vespalib/stllike/hash_map.h>
#include <vespa/vespalib/stllike/hash_map.hpp>
#include <vespa/vdslib/state/clusterstate.h>
#include <vespa/vdslib/state/cluster_state_bundle.h>
#include <vespa/fnet/frt/target.h>
#include <vespa/slobrok/sbmirror.h>
#include <getopt.h>
#include <iostream>

#include <vespa/log/log.h>
LOG_SETUP("vespa-spi-feed-bm");

using namespace config;
using namespace proton;
using namespace cloud::config::filedistribution;
using namespace vespa::config::search::core;
using namespace vespa::config::search::summary;
using namespace vespa::config::search;
using namespace std::chrono_literals;
using vespa::config::content::core::BucketspacesConfig;
using vespa::config::content::core::BucketspacesConfigBuilder;
using vespa::config::content::StorDistributionConfigBuilder;
using vespa::config::content::StorFilestorConfigBuilder;
using vespa::config::content::PersistenceConfigBuilder;
using vespa::config::content::core::StorBouncerConfigBuilder;
using vespa::config::content::core::StorCommunicationmanagerConfigBuilder;
using vespa::config::content::core::StorBucketInitConfigBuilder;
using vespa::config::content::core::StorOpsloggerConfigBuilder;
using vespa::config::content::core::StorPrioritymappingConfigBuilder;
using vespa::config::content::LoadTypeConfigBuilder;
using vespa::config::content::UpgradingConfigBuilder;
using vespa::config::content::core::StorServerConfigBuilder;
using vespa::config::content::core::StorStatusConfigBuilder;
using vespa::config::content::core::StorVisitorConfigBuilder;
using metrics::MetricsmanagerConfigBuilder;
using cloud::config::SlobroksConfigBuilder;
using messagebus::MessagebusConfigBuilder;

using config::ConfigContext;
using config::ConfigUri;
using config::ConfigSet;
using config::IConfigContext;
using document::AssignValueUpdate;
using document::BucketId;
using document::BucketSpace;
using document::Document;
using document::DocumentId;
using document::DocumentType;
using document::DocumentTypeRepo;
using document::DocumentTypeRepoFactory;
using document::DocumenttypesConfig;
using document::DocumenttypesConfigBuilder;
using document::DocumentUpdate;
using document::Field;
using document::FieldUpdate;
using document::IntFieldValue;
using document::test::makeBucketSpace;
using documentapi::LoadTypeSet;
using search::TuneFileDocumentDB;
using search::index::DummyFileHeaderContext;
using search::index::Schema;
using search::index::SchemaBuilder;
using search::transactionlog::TransLogServer;
using storage::rpc::MessageCodecProvider;
using storage::rpc::SharedRpcResources;
using storage::rpc::StorageApiRpcService;
using storage::spi::Bucket;
using storage::spi::PartitionId;
using storage::spi::PersistenceProvider;
using storage::spi::Priority;
using storage::spi::Timestamp;
using storage::spi::Trace;
using vespalib::makeLambdaTask;

using DocumentDBMap = std::map<DocTypeName, std::shared_ptr<DocumentDB>>;

namespace {

storage::spi::LoadType default_load_type(0, "default");

vespalib::string base_dir = "testdb";

std::shared_ptr<DocumenttypesConfig> make_document_type() {
    using Struct = document::config_builder::Struct;
    using DataType = document::DataType;
    document::config_builder::DocumenttypesConfigBuilderHelper builder;
    builder.document(42, "test", Struct("test.header").addField("int", DataType::T_INT), Struct("test.body"));
    return std::make_shared<DocumenttypesConfig>(builder.config());
}

std::shared_ptr<AttributesConfig> make_attributes_config() {
    AttributesConfigBuilder builder;
    AttributesConfig::Attribute attribute;
    attribute.name = "int";
    attribute.datatype = AttributesConfig::Attribute::Datatype::INT32;
    builder.attribute.emplace_back(attribute);
    return std::make_shared<AttributesConfig>(builder);
}

std::shared_ptr<DocumentDBConfig> make_document_db_config(std::shared_ptr<DocumenttypesConfig> document_types, std::shared_ptr<const DocumentTypeRepo> repo, const DocTypeName& doc_type_name)
{
    auto indexschema = std::make_shared<IndexschemaConfig>();
    auto attributes = make_attributes_config();
    auto summary = std::make_shared<SummaryConfig>();
    std::shared_ptr<Schema> schema(new Schema());
    SchemaBuilder::build(*indexschema, *schema);
    SchemaBuilder::build(*attributes, *schema);
    SchemaBuilder::build(*summary, *schema);
    return std::make_shared<DocumentDBConfig>(
            1,
            std::make_shared<RankProfilesConfig>(),
            std::make_shared<matching::RankingConstants>(),
            std::make_shared<matching::OnnxModels>(),
            indexschema,
            attributes,
            summary,
            std::make_shared<SummarymapConfig>(),
            std::make_shared<JuniperrcConfig>(),
            document_types,
            repo,
            std::make_shared<ImportedFieldsConfig>(),
            std::make_shared<TuneFileDocumentDB>(),
            schema,
            std::make_shared<DocumentDBMaintenanceConfig>(),
            search::LogDocumentStore::Config(),
            "client",
            doc_type_name.getName());
}

class MyPersistenceEngineOwner : public IPersistenceEngineOwner
{
    void setClusterState(BucketSpace, const storage::spi::ClusterState &) override { }
};

struct MyResourceWriteFilter : public IResourceWriteFilter
{
    bool acceptWriteOperation() const override { return true; }
    State getAcceptState() const override { return IResourceWriteFilter::State(); }
};

class MyPendingTracker {
    uint32_t                _pending;
    uint32_t                _limit;
    std::mutex              _mutex;
    std::condition_variable _cond;

public:
    MyPendingTracker(uint32_t limit)
        : _pending(0u),
          _limit(limit),
          _mutex(),
          _cond()
    {
    }

    ~MyPendingTracker()
    {
        drain();
    }

    void release() {
        std::unique_lock<std::mutex> guard(_mutex);
        --_pending;
        if (_pending < _limit) {
            _cond.notify_all();
        }
        //LOG(info, "release, pending is now %u", _pending);
    }
    void retain() {
        std::unique_lock<std::mutex> guard(_mutex);
        while (_pending >= _limit) {
            _cond.wait(guard);
        }
        ++_pending;
        //LOG(info, "retain, pending is now %u", _pending);
    }

    void drain() {
        std::unique_lock<std::mutex> guard(_mutex);
        while (_pending > 0) {
            _cond.wait(guard);
        }
    }
};

class MyOperationComplete : public storage::spi::OperationComplete
{
    MyPendingTracker& _tracker;
public:
    MyOperationComplete(MyPendingTracker &tracker);
    ~MyOperationComplete();
    void onComplete(std::unique_ptr<storage::spi::Result> result) override;
    void addResultHandler(const storage::spi::ResultHandler* resultHandler) override;
};

MyOperationComplete::MyOperationComplete(MyPendingTracker& tracker)
    : _tracker(tracker)
{
    _tracker.retain();
}

MyOperationComplete::~MyOperationComplete()
{
    _tracker.release();
}

void
MyOperationComplete::onComplete(std::unique_ptr<storage::spi::Result> result)
{
    (void) result;
}

void
MyOperationComplete::addResultHandler(const storage::spi::ResultHandler * resultHandler)
{
    (void) resultHandler;
}

class BMRange
{
    uint32_t _start;
    uint32_t _end;
public:
    BMRange(uint32_t start_in, uint32_t end_in)
        : _start(start_in),
          _end(end_in)
    {
    }
    uint32_t get_start() const { return _start; }
    uint32_t get_end() const { return _end; }
};

class BMParams {
    uint32_t _documents;
    uint32_t _threads;
    uint32_t _put_passes;
    uint32_t _update_passes;
    uint32_t _remove_passes;
    uint32_t _rpc_network_threads;
    bool     _enable_service_layer;
    uint32_t get_start(uint32_t thread_id) const {
        return (_documents / _threads) * thread_id + std::min(thread_id, _documents % _threads);
    }
public:
    BMParams()
        : _documents(160000),
          _threads(32),
          _put_passes(2),
          _update_passes(1),
          _remove_passes(2),
          _rpc_network_threads(1),
          _enable_service_layer(false)
    {
    }
    BMRange get_range(uint32_t thread_id) const {
        return BMRange(get_start(thread_id), get_start(thread_id + 1));
    }
    uint32_t get_documents() const { return _documents; }
    uint32_t get_threads() const { return _threads; }
    uint32_t get_put_passes() const { return _put_passes; }
    uint32_t get_update_passes() const { return _update_passes; }
    uint32_t get_remove_passes() const { return _remove_passes; }
    uint32_t get_rpc_network_threads() const { return _rpc_network_threads; }
    bool get_enable_service_layer() const { return _enable_service_layer; }
    void set_documents(uint32_t documents_in) { _documents = documents_in; }
    void set_threads(uint32_t threads_in) { _threads = threads_in; }
    void set_put_passes(uint32_t put_passes_in) { _put_passes = put_passes_in; }
    void set_update_passes(uint32_t update_passes_in) { _update_passes = update_passes_in; }
    void set_remove_passes(uint32_t remove_passes_in) { _remove_passes = remove_passes_in; }
    void set_rpc_network_threads(uint32_t threads_in) { _rpc_network_threads = threads_in; }
    void set_enable_service_layer(bool enable_service_layer_in) { _enable_service_layer = enable_service_layer_in; }
    bool check() const;
};

bool
BMParams::check() const
{
    if (_threads < 1) {
        std::cerr << "Too few threads: " << _threads << std::endl;
        return false;
    }
    if (_threads > 256) {
        std::cerr << "Too many threads: " << _threads << std::endl;
        return false;
    }
    if (_documents < _threads) {
        std::cerr << "Too few documents: " << _documents << std::endl;
        return false;
    }
    if (_put_passes < 1) {
        std::cerr << "Put passes too low: " << _put_passes << std::endl;
        return false;
    }
    if (_rpc_network_threads < 1) {
        std::cerr << "Too few rpc network threads: " << _rpc_network_threads << std::endl;
        return false;
    }
    return true;
}


class MyServiceLayerProcess : public storage::ServiceLayerProcess {
    PersistenceProvider&    _provider;

public:
    MyServiceLayerProcess(const config::ConfigUri & configUri,
                          PersistenceProvider &provider);
    ~MyServiceLayerProcess() override { shutdown(); }

    void shutdown() override;
    void setupProvider() override;
    PersistenceProvider& getProvider() override;
};

MyServiceLayerProcess::MyServiceLayerProcess(const config::ConfigUri & configUri,
                                             PersistenceProvider &provider)
    : ServiceLayerProcess(configUri),
      _provider(provider)
{
}

void
MyServiceLayerProcess::shutdown()
{
    ServiceLayerProcess::shutdown();
}

void
MyServiceLayerProcess::setupProvider()
{
}

PersistenceProvider&
MyServiceLayerProcess::getProvider()
{
    return _provider;
}

struct MyStorageConfig
{
    vespalib::string              config_id;
    DocumenttypesConfigBuilder    documenttypes;
    PersistenceConfigBuilder      persistence;
    StorDistributionConfigBuilder stor_distribution;
    StorFilestorConfigBuilder     stor_filestor;
    StorBouncerConfigBuilder      stor_bouncer;
    StorCommunicationmanagerConfigBuilder stor_communicationmanager;
    StorBucketInitConfigBuilder   stor_bucket_init;
    StorOpsloggerConfigBuilder    stor_opslogger;
    StorPrioritymappingConfigBuilder stor_prioritymapping;
    UpgradingConfigBuilder        upgrading;
    StorServerConfigBuilder       stor_server;
    StorStatusConfigBuilder       stor_status;
    StorVisitorConfigBuilder      stor_visitor;
    BucketspacesConfigBuilder     bucketspaces;
    LoadTypeConfigBuilder         load_type;
    MetricsmanagerConfigBuilder   metricsmanager;
    SlobroksConfigBuilder         slobroks;
    MessagebusConfigBuilder       messagebus;

    MyStorageConfig(const vespalib::string& config_id_in, const DocumenttypesConfig& documenttypes_in, int slobrok_port, int status_port, uint32_t rpc_network_threads)
        : config_id(config_id_in),
          documenttypes(documenttypes_in),
          persistence(),
          stor_distribution(),
          stor_filestor(),
          stor_bouncer(),
          stor_communicationmanager(),
          stor_bucket_init(),
          stor_opslogger(),
          stor_prioritymapping(),
          upgrading(),
          stor_server(),
          stor_status(),
          stor_visitor(),
          bucketspaces(),
          load_type(),
          metricsmanager(),
          slobroks(),
          messagebus()
    {
        {
            auto &dc = stor_distribution;
            {
                StorDistributionConfigBuilder::Group group;
                {
                    StorDistributionConfigBuilder::Group::Nodes node;
                    node.index = 0;
                    group.nodes.push_back(std::move(node));
                }
                group.index = "invalid";
                group.name = "invalid";
                group.capacity = 1.0;
                group.partitions = "";
                dc.group.push_back(std::move(group));
            }
            dc.redundancy = 1;
            dc.readyCopies = 1;
        }
        stor_server.rootFolder = "storage";
        {
            SlobroksConfigBuilder::Slobrok slobrok;
            slobrok.connectionspec = vespalib::make_string("tcp/localhost:%d", slobrok_port);
            slobroks.slobrok.push_back(std::move(slobrok));
        }
        stor_communicationmanager.useDirectStorageapiRpc = true;
        stor_communicationmanager.rpc.numNetworkThreads = rpc_network_threads;
        stor_status.httpport = status_port;
    }

    ~MyStorageConfig();

    void add_builders(ConfigSet &set) {
        set.addBuilder(config_id, &documenttypes);
        set.addBuilder(config_id, &persistence);
        set.addBuilder(config_id, &stor_distribution);
        set.addBuilder(config_id, &stor_filestor);
        set.addBuilder(config_id, &stor_bouncer);
        set.addBuilder(config_id, &stor_communicationmanager);
        set.addBuilder(config_id, &stor_bucket_init);
        set.addBuilder(config_id, &stor_opslogger);
        set.addBuilder(config_id, &stor_prioritymapping);
        set.addBuilder(config_id, &upgrading);
        set.addBuilder(config_id, &stor_server);
        set.addBuilder(config_id, &stor_status);
        set.addBuilder(config_id, &stor_visitor);
        set.addBuilder(config_id, &bucketspaces);
        set.addBuilder(config_id, &load_type);
        set.addBuilder(config_id, &metricsmanager);
        set.addBuilder(config_id, &slobroks);
        set.addBuilder(config_id, &messagebus);
    }
};

MyStorageConfig::~MyStorageConfig() = default;

struct MyRpcClientConfig {
    vespalib::string      config_id;
    SlobroksConfigBuilder slobroks;

    MyRpcClientConfig(const vespalib::string &config_id_in, int slobrok_port)
        : config_id(config_id_in),
          slobroks()
    {
        {
            SlobroksConfigBuilder::Slobrok slobrok;
            slobrok.connectionspec = vespalib::make_string("tcp/localhost:%d", slobrok_port);
            slobroks.slobrok.push_back(std::move(slobrok));
        }
    }
    ~MyRpcClientConfig();

    void add_builders(ConfigSet &set) {
        set.addBuilder(config_id, &slobroks);
    }
};

MyRpcClientConfig::~MyRpcClientConfig() = default;

class MyMessageDispatcher : public storage::MessageDispatcher
{
    std::mutex _mutex;
    vespalib::hash_map<uint64_t, MyPendingTracker *> _pending;
public:
    MyMessageDispatcher()
        : storage::MessageDispatcher(),
          _mutex(),
          _pending()
    {
    }
    ~MyMessageDispatcher() override;
    void dispatch_sync(std::shared_ptr<storage::api::StorageMessage> msg) override {
        release(msg->getMsgId());
    }
    void dispatch_async(std::shared_ptr<storage::api::StorageMessage> msg) override {
        release(msg->getMsgId());
    }
    void retain(uint64_t msg_id, MyPendingTracker &tracker) {
        tracker.retain();
        std::lock_guard lock(_mutex);
        _pending.insert(std::make_pair(msg_id, &tracker));
    }
    void release(uint64_t msg_id) {
        MyPendingTracker *tracker = nullptr;
        {
            std::lock_guard lock(_mutex);
            auto itr = _pending.find(msg_id);
            assert(itr != _pending.end());
            tracker = itr->second;
            _pending.erase(itr);
        }
        tracker->release();
    }
};

MyMessageDispatcher::~MyMessageDispatcher()
{
    std::lock_guard lock(_mutex);
    assert(_pending.empty());
}

FRT_RPCRequest *make_set_cluster_state_request() {
    storage::lib::ClusterStateBundle bundle(storage::lib::ClusterState("version:2 distributor:1 storage:1"));
    storage::rpc::SlimeClusterStateBundleCodec codec;
    auto encoded_bundle = codec.encode(bundle);
    auto *req = new FRT_RPCRequest();
    auto* params = req->GetParams();
    params->AddInt8(static_cast<uint8_t>(encoded_bundle._compression_type));
    params->AddInt32(encoded_bundle._uncompressed_length);
    const auto buf_len = encoded_bundle._buffer->getDataLen();
    params->AddData(encoded_bundle._buffer->stealBuffer(), buf_len);
    req->SetMethodName("setdistributionstates");
    return req;
}

void set_cluster_up(SharedRpcResources &shared_rpc_resources, storage::api::StorageMessageAddress &storage_address) {
    auto req = make_set_cluster_state_request();
    auto target_resolver = std::make_unique<storage::rpc::CachingRpcTargetResolver>(shared_rpc_resources.slobrok_mirror(), shared_rpc_resources.target_factory());
    auto target = target_resolver->resolve_rpc_target(storage_address);
    target->_target->get()->InvokeSync(req, 10.0); // 10 seconds timeout
    assert(!req->IsError());
    req->SubRef();
}

}

struct PersistenceProviderFixture {
    std::shared_ptr<DocumenttypesConfig>       _document_types;
    std::shared_ptr<const DocumentTypeRepo>    _repo;
    DocTypeName                                _doc_type_name;
    const DocumentType*                        _document_type;
    const Field&                               _field;
    std::shared_ptr<DocumentDBConfig>          _document_db_config;
    vespalib::string                           _base_dir;
    DummyFileHeaderContext                     _file_header_context;
    int                                        _tls_listen_port;
    int                                        _slobrok_port;
    int                                        _status_port;
    int                                        _rpc_client_port;
    TransLogServer                             _tls;
    vespalib::string                           _tls_spec;
    matching::QueryLimiter                     _query_limiter;
    vespalib::Clock                            _clock;
    DummyWireService                           _metrics_wire_service;
    MemoryConfigStores                         _config_stores;
    vespalib::ThreadStackExecutor              _summary_executor;
    DummyDBOwner                               _document_db_owner;
    BucketSpace                                _bucket_space;
    std::shared_ptr<DocumentDB>                _document_db;
    MyPersistenceEngineOwner                   _persistence_owner;
    MyResourceWriteFilter                      _write_filter;
    std::shared_ptr<PersistenceEngine>         _persistence_engine;
    storage::spi::Context                      _context;
    uint32_t                                   _bucket_bits;
    MyStorageConfig                            _service_layer_config;
    MyRpcClientConfig                          _rpc_client_config;
    ConfigSet                                  _config_set;
    std::shared_ptr<IConfigContext>            _config_context;
    storage::api::StorageMessageAddress        _storage_address;
    std::unique_ptr<mbus::Slobrok>             _slobrok;
    std::unique_ptr<MyServiceLayerProcess>     _service_layer;
    std::unique_ptr<MessageCodecProvider>      _message_codec_provider;
    std::unique_ptr<SharedRpcResources>        _rpc_client_shared_rpc_resources;
    std::unique_ptr<MyMessageDispatcher>       _rpc_message_dispatcher;
    std::unique_ptr<StorageApiRpcService>      _rpc_client;

    PersistenceProviderFixture(const BMParams& params);
    ~PersistenceProviderFixture();
    void create_document_db();
    uint32_t num_buckets() const { return (1u << _bucket_bits); }
    BucketId make_bucket_id(uint32_t i) const { return BucketId(_bucket_bits, i & (num_buckets() - 1)); }
    Bucket make_bucket(uint32_t i) const { return Bucket(document::Bucket(_bucket_space, BucketId(_bucket_bits, i & (num_buckets() - 1))), PartitionId(0)); }
    DocumentId make_document_id(uint32_t i) const;
    std::unique_ptr<Document> make_document(uint32_t i) const;
    std::unique_ptr<DocumentUpdate> make_document_update(uint32_t i) const;
    void create_buckets();
    void start_service_layer();
    void shutdown_service_layer();
    void send_rpc(std::shared_ptr<storage::api::StorageCommand> cmd, MyPendingTracker& pending_tracker);
};

PersistenceProviderFixture::PersistenceProviderFixture(const BMParams& params)
    : _document_types(make_document_type()),
      _repo(DocumentTypeRepoFactory::make(*_document_types)),
      _doc_type_name("test"),
      _document_type(_repo->getDocumentType(_doc_type_name.getName())),
      _field(_document_type->getField("int")),
      _document_db_config(make_document_db_config(_document_types, _repo, _doc_type_name)),
      _base_dir(base_dir),
      _file_header_context(),
      _tls_listen_port(9017),
      _slobrok_port(9018),
      _status_port(9019),
      _rpc_client_port(9020),
      _tls("tls", _tls_listen_port, _base_dir, _file_header_context),
      _tls_spec(vespalib::make_string("tcp/localhost:%d", _tls_listen_port)),
      _query_limiter(),
      _clock(),
      _metrics_wire_service(),
      _config_stores(),
      _summary_executor(8, 128 * 1024),
      _document_db_owner(),
      _bucket_space(makeBucketSpace(_doc_type_name.getName())),
      _document_db(),
      _persistence_owner(),
      _write_filter(),
      _persistence_engine(),
      _context(default_load_type, Priority(0), Trace::TraceLevel(0)),
      _bucket_bits(16),
      _service_layer_config("bm-servicelayer", *_document_types, _slobrok_port, _status_port, params.get_rpc_network_threads()),
      _rpc_client_config("bm-rpc-client", _slobrok_port),
      _config_set(),
      _config_context(std::make_shared<ConfigContext>(_config_set)),
      _storage_address("storage", storage::lib::NodeType::STORAGE, 0),
      _slobrok(),
      _service_layer(),
      _message_codec_provider(),
      _rpc_client_shared_rpc_resources(),
      _rpc_message_dispatcher(),
      _rpc_client()
{
    create_document_db();
    _persistence_engine = std::make_unique<PersistenceEngine>(_persistence_owner, _write_filter, -1, false);
    auto proxy = std::make_shared<PersistenceHandlerProxy>(_document_db);
    _persistence_engine->putHandler(_persistence_engine->getWLock(), _bucket_space, _doc_type_name, proxy);
    _service_layer_config.add_builders(_config_set);
    _rpc_client_config.add_builders(_config_set);
}

PersistenceProviderFixture::~PersistenceProviderFixture()
{
    if (_persistence_engine) {
        _persistence_engine->destroyIterators();
        _persistence_engine->removeHandler(_persistence_engine->getWLock(), _bucket_space, _doc_type_name);
    }
    if (_document_db) {
        _document_db->close();
    }
}

void
PersistenceProviderFixture::create_document_db()
{
    vespalib::mkdir(_base_dir, false);
    vespalib::mkdir(_base_dir + "/" + _doc_type_name.getName(), false);
    vespalib::string input_cfg = _base_dir + "/" + _doc_type_name.getName() + "/baseconfig";
    {
        FileConfigManager fileCfg(input_cfg, "", _doc_type_name.getName());
        fileCfg.saveConfig(*_document_db_config, 1);
    }
    config::DirSpec spec(input_cfg + "/config-1");
    auto tuneFileDocDB = std::make_shared<TuneFileDocumentDB>();
    DocumentDBConfigHelper mgr(spec, _doc_type_name.getName());
    auto bootstrap_config = std::make_shared<BootstrapConfig>(1,
                                                              _document_types,
                                                              _repo,
                                                              std::make_shared<ProtonConfig>(),
                                                              std::make_shared<FiledistributorrpcConfig>(),
                                                              std::make_shared<BucketspacesConfig>(),
                                                              tuneFileDocDB, HwInfo());
    mgr.forwardConfig(bootstrap_config);
    mgr.nextGeneration(0ms);
    _document_db = std::make_shared<DocumentDB>(_base_dir,
                                                mgr.getConfig(),
                                                _tls_spec,
                                                _query_limiter,
                                                _clock,
                                                _doc_type_name,
                                                _bucket_space,
                                                *bootstrap_config->getProtonConfigSP(),
                                                _document_db_owner,
                                                _summary_executor,
                                                _summary_executor,
                                                _tls,
                                                _metrics_wire_service,
                                                _file_header_context,
                                                _config_stores.getConfigStore(_doc_type_name.toString()),
                                                std::make_shared<vespalib::ThreadStackExecutor>(16, 128 * 1024),
                                                HwInfo());
    _document_db->start();
    _document_db->waitForOnlineState();
}

DocumentId
PersistenceProviderFixture::make_document_id(uint32_t i) const
{
    DocumentId id(vespalib::make_string("id::test:n=%u:%u", i & (num_buckets() - 1), i));
    return id;
}

std::unique_ptr<Document>
PersistenceProviderFixture::make_document(uint32_t i) const
{
    auto id = make_document_id(i);
    auto document = std::make_unique<Document>(*_document_type, id);
    document->setRepo(*_repo);
    document->setFieldValue(_field, std::make_unique<IntFieldValue>(i));
    return document;
}

std::unique_ptr<DocumentUpdate>
PersistenceProviderFixture::make_document_update(uint32_t i) const
{
    auto id = make_document_id(i);
    auto document_update = std::make_unique<DocumentUpdate>(*_repo, *_document_type, id);
    document_update->addUpdate(FieldUpdate(_field).addUpdate(AssignValueUpdate(IntFieldValue(15))));
    return document_update;
}

void
PersistenceProviderFixture::create_buckets()
{
    auto &provider = *_persistence_engine;
    for (unsigned int i = 0; i < num_buckets(); ++i) {
        provider.createBucket(make_bucket(i), _context);
    }
}

void
PersistenceProviderFixture::start_service_layer()
{
    LOG(info, "start slobrok");
    _slobrok = std::make_unique<mbus::Slobrok>(_slobrok_port);
    LOG(info, "start service layer");
    config::ConfigUri config_uri("bm-servicelayer", _config_context);
    _service_layer = std::make_unique<MyServiceLayerProcess>(config_uri,
                                                             *_persistence_engine);
    _service_layer->setupConfig(100ms);
    _service_layer->createNode();
    _service_layer->getNode().waitUntilInitialized();
    _message_codec_provider = std::make_unique<MessageCodecProvider>(_repo, std::make_shared<documentapi::LoadTypeSet>());
    LOG(info, "start rpc client shared resources");
    config::ConfigUri client_config_uri("bm-rpc-client", _config_context);
    _rpc_client_shared_rpc_resources = std::make_unique<SharedRpcResources>(client_config_uri, _rpc_client_port, 100);
    _rpc_client_shared_rpc_resources->start_server_and_register_slobrok("bm-rpc-client");
    _rpc_message_dispatcher = std::make_unique<MyMessageDispatcher>();
    _rpc_client = std::make_unique<StorageApiRpcService>(*_rpc_message_dispatcher, *_rpc_client_shared_rpc_resources, *_message_codec_provider, StorageApiRpcService::Params());
    set_cluster_up(*_rpc_client_shared_rpc_resources, _storage_address);
}

void
PersistenceProviderFixture::shutdown_service_layer()
{
    _rpc_client.reset();
    _rpc_message_dispatcher.reset();
    if (_rpc_client_shared_rpc_resources) {
        LOG(info, "stop rpc client shared resources");
        _rpc_client_shared_rpc_resources->shutdown();
        _rpc_client_shared_rpc_resources.reset();
    }
    if (_service_layer) {
        LOG(info, "stop service layer");
        _service_layer->getNode().requestShutdown("controlled shutdown");
        _service_layer->shutdown();
    }
    if (_slobrok) {
        LOG(info, "stop slobrok");
        _slobrok.reset();
    }
}

void
PersistenceProviderFixture::send_rpc(std::shared_ptr<storage::api::StorageCommand> cmd, MyPendingTracker& pending_tracker)
{
    cmd->setSourceIndex(0);
    cmd->setAddress(_storage_address);
    _rpc_message_dispatcher->retain(cmd->getMsgId(), pending_tracker);
    _rpc_client->send_rpc_v1_request(std::move(cmd));
}

vespalib::nbostream
make_put_feed(PersistenceProviderFixture &f, BMRange range)
{
    vespalib::nbostream serialized_feed;
    LOG(debug, "make_put_feed([%u..%u))", range.get_start(), range.get_end());
    for (unsigned int i = range.get_start(); i < range.get_end(); ++i) {
        serialized_feed << f.make_bucket_id(i);
        auto document = f.make_document(i);
        document->serialize(serialized_feed);
    }
    return serialized_feed;
}

std::vector<vespalib::nbostream>
make_feed(vespalib::ThreadStackExecutor &executor, const BMParams &bm_params, std::function<vespalib::nbostream(BMRange)> func, const vespalib::string &label)
{
    LOG(info, "make_feed %s %u small documents", label.c_str(), bm_params.get_documents());
    std::vector<vespalib::nbostream> serialized_feed_v;
    auto start_time = std::chrono::steady_clock::now();
    serialized_feed_v.resize(bm_params.get_threads());
    for (uint32_t i = 0; i < bm_params.get_threads(); ++i) {
        auto range = bm_params.get_range(i);
        executor.execute(makeLambdaTask([&serialized_feed_v, i, range, &func]()
                                        { serialized_feed_v[i] = func(range); }));
    }
    executor.sync();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    LOG(info, "%8.2f %s data elements/s", bm_params.get_documents() / elapsed.count(), label.c_str());
    return serialized_feed_v;
}

void
put_async_task(PersistenceProviderFixture &f, BMRange range, const vespalib::nbostream &serialized_feed, int64_t time_bias)
{
    LOG(debug, "put_async_task([%u..%u))", range.get_start(), range.get_end());
    MyPendingTracker pending_tracker(100);
    auto &provider = *f._persistence_engine;
    auto &context = f._context;
    auto &repo = *f._repo;
    vespalib::nbostream is(serialized_feed.data(), serialized_feed.size());
    BucketId bucket_id;
    auto bucket_space = f._bucket_space;
    for (unsigned int i = range.get_start(); i < range.get_end(); ++i) {
        is >> bucket_id;
        Bucket bucket(document::Bucket(bucket_space, bucket_id), PartitionId(0));
        auto document = std::make_unique<Document>(repo, is);
        if (f._rpc_client) {
            auto cmd = std::make_unique<storage::api::PutCommand>(bucket.getBucket(), std::move(document), time_bias + i);
            f.send_rpc(std::move(cmd), pending_tracker);
        } else {
            provider.putAsync(bucket, Timestamp(time_bias + i), std::move(document), context, std::make_unique<MyOperationComplete>(pending_tracker));
        }
    }
    assert(is.empty());
    pending_tracker.drain();
}

void
run_put_async_tasks(PersistenceProviderFixture &f, vespalib::ThreadStackExecutor &executor, int pass, int64_t& time_bias, const std::vector<vespalib::nbostream> &serialized_feed_v, const BMParams& bm_params)
{
    LOG(info, "putAsync %u small documents, pass=%u", bm_params.get_documents(), pass);
    auto start_time = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < bm_params.get_threads(); ++i) {
        auto range = bm_params.get_range(i);
        executor.execute(makeLambdaTask([&f, &serialized_feed = serialized_feed_v[i], range, time_bias]()
                                        { put_async_task(f, range, serialized_feed, time_bias); }));
    }
    executor.sync();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    LOG(info, "%8.2f puts/s for pass=%u", bm_params.get_documents() / elapsed.count(), pass);
    time_bias += bm_params.get_documents();
}

vespalib::nbostream
make_update_feed(PersistenceProviderFixture &f, BMRange range)
{
    vespalib::nbostream serialized_feed;
    LOG(debug, "make_update_feed([%u..%u))", range.get_start(), range.get_end());
    for (unsigned int i = range.get_start(); i < range.get_end(); ++i) {
        serialized_feed << f.make_bucket_id(i);
        auto document_update = f.make_document_update(i);
        document_update->serializeHEAD(serialized_feed);
    }
    return serialized_feed;
}

void
update_async_task(PersistenceProviderFixture &f, BMRange range, const vespalib::nbostream &serialized_feed, int64_t time_bias)
{
    LOG(debug, "update_async_task([%u..%u))", range.get_start(), range.get_end());
    MyPendingTracker pending_tracker(100);
    auto &provider = *f._persistence_engine;
    auto &context = f._context;
    auto &repo = *f._repo;
    vespalib::nbostream is(serialized_feed.data(), serialized_feed.size());
    BucketId bucket_id;
    auto bucket_space = f._bucket_space;
    for (unsigned int i = range.get_start(); i < range.get_end(); ++i) {
        is >> bucket_id;
        Bucket bucket(document::Bucket(bucket_space, bucket_id), PartitionId(0));
        auto document_update = DocumentUpdate::createHEAD(repo, is);
        if (f._rpc_client) {
            auto cmd = std::make_unique<storage::api::UpdateCommand>(bucket.getBucket(), std::move(document_update), time_bias + i);
            f.send_rpc(std::move(cmd), pending_tracker);
        } else {
            provider.updateAsync(bucket, Timestamp(time_bias + i), std::move(document_update), context, std::make_unique<MyOperationComplete>(pending_tracker));
        }
    }
    assert(is.empty());
    pending_tracker.drain();
}

void
run_update_async_tasks(PersistenceProviderFixture &f, vespalib::ThreadStackExecutor &executor, int pass, int64_t& time_bias, const std::vector<vespalib::nbostream> &serialized_feed_v, const BMParams& bm_params)
{
    LOG(info, "updateAsync %u small documents, pass=%u", bm_params.get_documents(), pass);
    auto start_time = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < bm_params.get_threads(); ++i) {
        auto range = bm_params.get_range(i);
        executor.execute(makeLambdaTask([&f, &serialized_feed = serialized_feed_v[i], range, time_bias]()
                                        { update_async_task(f, range, serialized_feed, time_bias); }));
    }
    executor.sync();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    LOG(info, "%8.2f updates/s for pass=%u", bm_params.get_documents() / elapsed.count(), pass);
    time_bias += bm_params.get_documents();
}

vespalib::nbostream
make_remove_feed(PersistenceProviderFixture &f, BMRange range)
{
    vespalib::nbostream serialized_feed;
    LOG(debug, "make_update_feed([%u..%u))", range.get_start(), range.get_end());
    for (unsigned int i = range.get_start(); i < range.get_end(); ++i) {
        serialized_feed << f.make_bucket_id(i);
        auto document_id = f.make_document_id(i);
        vespalib::string raw_id = document_id.toString();
        serialized_feed.write(raw_id.c_str(), raw_id.size() + 1);
    }
    return serialized_feed;
}

void
remove_async_task(PersistenceProviderFixture &f, BMRange range, const vespalib::nbostream &serialized_feed, int64_t time_bias)
{
    LOG(debug, "remove_async_task([%u..%u))", range.get_start(), range.get_end());
    MyPendingTracker pending_tracker(100);
    auto &provider = *f._persistence_engine;
    auto &context = f._context;
    vespalib::nbostream is(serialized_feed.data(), serialized_feed.size());
    BucketId bucket_id;
    auto bucket_space = f._bucket_space;
    for (unsigned int i = range.get_start(); i < range.get_end(); ++i) {
        is >> bucket_id;
        Bucket bucket(document::Bucket(bucket_space, bucket_id), PartitionId(0));
        DocumentId document_id(is);
        if (f._rpc_client) {
            auto cmd = std::make_unique<storage::api::RemoveCommand>(bucket.getBucket(), document_id, time_bias + i);
            f.send_rpc(std::move(cmd), pending_tracker);
        } else {
            provider.removeAsync(bucket, Timestamp(time_bias + i), document_id, context, std::make_unique<MyOperationComplete>(pending_tracker));
        }
    }
    assert(is.empty());
    pending_tracker.drain();
}

void
run_remove_async_tasks(PersistenceProviderFixture &f, vespalib::ThreadStackExecutor &executor, int pass, int64_t& time_bias, const std::vector<vespalib::nbostream> &serialized_feed_v, const BMParams &bm_params)
{
    LOG(info, "removeAsync %u small documents, pass=%u", bm_params.get_documents(), pass);
    auto start_time = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < bm_params.get_threads(); ++i) {
        auto range = bm_params.get_range(i);
        executor.execute(makeLambdaTask([&f, &serialized_feed = serialized_feed_v[i], range, time_bias]()
                                        { remove_async_task(f, range, serialized_feed, time_bias); }));
    }
    executor.sync();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    LOG(info, "%8.2f removes/s for pass=%u", bm_params.get_documents() / elapsed.count(), pass);
    time_bias += bm_params.get_documents();
}

void benchmark_async_spi(const BMParams &bm_params)
{
    vespalib::rmdir(base_dir, true);
    PersistenceProviderFixture f(bm_params);
    auto &provider = *f._persistence_engine;
    LOG(info, "start initialize");
    provider.initialize();
    LOG(info, "create %u buckets", f.num_buckets());
    f.create_buckets();
    if (bm_params.get_enable_service_layer()) {
        f.start_service_layer();
    }
    vespalib::ThreadStackExecutor executor(bm_params.get_threads(), 128 * 1024);
    auto put_feed = make_feed(executor, bm_params, [&f](BMRange range) { return make_put_feed(f, range); }, "put");
    auto update_feed = make_feed(executor, bm_params, [&f](BMRange range) { return make_update_feed(f, range); }, "update");
    auto remove_feed = make_feed(executor, bm_params, [&f](BMRange range) { return make_remove_feed(f, range); }, "remove");
    int64_t time_bias = 1;
    for (uint32_t pass = 0; pass < bm_params.get_put_passes(); ++pass) {
        run_put_async_tasks(f, executor, pass, time_bias, put_feed, bm_params);
    }
    for (uint32_t pass = 0; pass < bm_params.get_update_passes(); ++pass) {
        run_update_async_tasks(f, executor, pass, time_bias, update_feed, bm_params);
    }
    for (uint32_t pass = 0; pass < bm_params.get_remove_passes(); ++pass) {
        run_remove_async_tasks(f, executor, pass, time_bias, remove_feed, bm_params);
    }
    f.shutdown_service_layer();
}

class App : public FastOS_Application
{
    BMParams _bm_params;
public:
    App();
    ~App() override;
    void usage();
    bool get_options();
    int Main() override;
};

App::App()
    : _bm_params()
{
}

App::~App() = default;

void
App::usage()
{
    std::cerr <<
        "vespa-spi-feed-bm version 0.0\n"
        "\n"
        "USAGE:\n";
    std::cerr <<
        "vespa-spi-feed-bm\n"
        "[--threads threads]\n"
        "[--documents documents]\n"
        "[--put-passes put-passes]\n"
        "[--update-passes update-passes]\n"
        "[--remove-passes remove-passes]\n"
        "[--rpc-network-threads threads]\n"
        "[--enable-service-layer]" << std::endl;
}

bool
App::get_options()
{
    int c;
    const char *opt_argument = nullptr;
    int long_opt_index = 0;
    static struct option long_opts[] = {
        { "threads", 1, nullptr, 0 },
        { "documents", 1, nullptr, 0 },
        { "put-passes", 1, nullptr, 0 },
        { "update-passes", 1, nullptr, 0 },
        { "remove-passes", 1, nullptr, 0 },
        { "rpc-network-threads", 1, nullptr, 0 },
        { "enable-service-layer", 0, nullptr, 0 }
    };
    enum longopts_enum {
        LONGOPT_THREADS,
        LONGOPT_DOCUMENTS,
        LONGOPT_PUT_PASSES,
        LONGOPT_UPDATE_PASSES,
        LONGOPT_REMOVE_PASSES,
        LONGOPT_RPC_NETWORK_THREADS,
        LONGOPT_ENABLE_SERVICE_LAYER
    };
    int opt_index = 1;
    resetOptIndex(opt_index);
    while ((c = GetOptLong("", opt_argument, opt_index, long_opts, &long_opt_index)) != -1) {
        switch (c) {
        case 0:
            switch(long_opt_index) {
            case LONGOPT_THREADS:
                _bm_params.set_threads(atoi(opt_argument));
                break;
            case LONGOPT_DOCUMENTS:
                _bm_params.set_documents(atoi(opt_argument));
                break;
            case LONGOPT_PUT_PASSES:
                _bm_params.set_put_passes(atoi(opt_argument));
                break;
            case LONGOPT_UPDATE_PASSES:
                _bm_params.set_update_passes(atoi(opt_argument));
                break;
            case LONGOPT_REMOVE_PASSES:
                _bm_params.set_remove_passes(atoi(opt_argument));
                break;
            case LONGOPT_RPC_NETWORK_THREADS:
                _bm_params.set_rpc_network_threads(atoi(opt_argument));
                break;
            case LONGOPT_ENABLE_SERVICE_LAYER:
                _bm_params.set_enable_service_layer(true);
                break;
            default:
                return false;
            }
            break;
        default:
            return false;
        }
    }
    return _bm_params.check();
}

int
App::Main()
{
    if (!get_options()) {
        usage();
        return 1;
    }
    benchmark_async_spi(_bm_params);
    return 0;
}

int
main(int argc, char* argv[])
{
    DummyFileHeaderContext::setCreator("vespa-spi-feed-bm");
    App app;
    auto exit_value = app.Entry(argc, argv);
    vespalib::rmdir(base_dir, true);
    return exit_value;
}
